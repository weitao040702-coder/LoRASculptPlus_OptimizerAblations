"""User-authorized migration of bulky experiment assets to /data0/weitaoma."""
from pathlib import Path
import json,os,signal,subprocess,time,shutil,sys
ROOT=Path('/home/weitaoma/LoRASculptPlus_OptimizerAblations')
DEST=Path('/data0/weitaoma/LoRASculptPlus_OptimizerAblations')
LOG=ROOT/'logs/storage-migration.log'
STATUS=ROOT/'logs/storage-migration-status.json'
JOBS=[(ROOT/'assets',DEST/'assets'),(ROOT/'checkpoints',DEST/'checkpoints'),(ROOT/'automation',DEST/'automation'),(ROOT/'.cache',DEST/'cache'),(Path('/home/weitaoma/.cache/pip'),Path('/data0/weitaoma/cache/pip'))]

def status(stage,**kwargs):
 d=dict(stage=stage,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));d.update(kwargs);tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(STATUS);print(json.dumps(d),flush=True)

def sync(src,dst,verify=False):
 dst.mkdir(parents=True,exist_ok=True)
 cmd=['rsync','-a','--delete','--exclude=tmp/***']
 if verify:cmd+=['--checksum','--dry-run','--itemize-changes']
 cmd += [str(src)+'/',str(dst)+'/']
 if verify:
  p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
  if p.stdout.strip():raise RuntimeError('Verification differs for '+str(src)+': '+p.stdout[:1500])
 else:subprocess.run(cmd,check=True)

def process_ids():
 wanted={b'scripts/auto_paper_pipeline.py',b'scripts/parallel_completed_eval.py',b'scripts/watch_paper_progress.py',b'scripts/resume_queued_rank32_eval.py'}
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if any(x in wanted for x in argv):rows.append(int(p.name))
  except (OSError,ValueError):pass
 return rows

def main():
 old=Path('/data0/weitaoma/language-following-model')
 status('DELETE_AUTHORIZED_OLD_PROJECT',path=str(old))
 if old.exists():
  assert not old.is_symlink() and old.resolve()==old and old.stat().st_uid==os.getuid()
  shutil.rmtree(old)
 assert not old.exists()
 status('COPYING_INITIAL',destination=str(DEST))
 for src,dst in JOBS:
  if src.exists():
   assert not src.is_symlink(),str(src)+' already a symlink'
   status('COPYING_INITIAL',source=str(src),destination=str(dst));sync(src,dst)
 status('STOPPING_FOR_FINAL_SYNC')
 for pid in process_ids():
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 for _ in range(20):
  if not process_ids():break
  time.sleep(1)
 assert not process_ids(),'Experiment schedulers did not stop'
 # Allow terminated workers to release output files before final copy.
 time.sleep(3)
 for src,dst in JOBS:
  if not src.exists():continue
  status('FINAL_SYNC',source=str(src));sync(src,dst)
  status('VERIFYING_CHECKSUMS',source=str(src));sync(src,dst,verify=True)
 status('SWITCHING_PATHS')
 for src,dst in JOBS:
  if not src.exists():continue
  backup=src.with_name(src.name+'.storage-migration-old')
  assert not backup.exists()
  src.rename(backup)
  try:src.symlink_to(dst,target_is_directory=True)
  except BaseException:backup.rename(src);raise
  shutil.rmtree(backup)
 (DEST/'cache/tmp').mkdir(parents=True,exist_ok=True)
 report=dict(destination=str(DEST),mappings={str(src):str(dst) for src,dst in JOBS},old_project_deleted=str(old),verification='rsync full content checksums before original removal',code_remains=str(ROOT),updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 (ROOT/'STORAGE_LAYOUT.json').write_text(json.dumps(report,indent=2)+'\n')
 status('RESTARTING_EXPERIMENTS')
 python=ROOT/'.venv_new_scheme/bin/python'
 pids={}
 for script,log in [('auto_paper_pipeline.py','pipeline.log'),('parallel_completed_eval.py','parallel-eval.log'),('watch_paper_progress.py','monitor.log')]:
  out=(ROOT/'logs/auto-paper'/log).open('a');p=subprocess.Popen([str(python),'-u','scripts/'+script],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);pids[script]=p.pid
 status('COMPLETED',**report,restarted_pids=pids)

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  status('FAILED',error=repr(exc));raise
