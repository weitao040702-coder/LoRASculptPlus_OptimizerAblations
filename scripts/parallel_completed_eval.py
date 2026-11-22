"""Evaluate completed ranks on GPU0 while the independent rank64 trainer runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from auto_paper_pipeline import ROOT, WORK, LOG, PYTHON, EVAL_FOLDER, environment, eval_ready, digest, atomic

ACTIVE = None
RANKS = [16,32]
STATUS_NAME = "parallel-eval-status.json"
LOCK_NAME = "parallel-eval.lock"

def status(stage, **extra):
    atomic(WORK/STATUS_NAME,dict(stage=stage,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu=0,**extra))

def stop(signum, frame):
    if ACTIVE is not None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
    raise SystemExit(128+signum)

def main():
    global ACTIVE
    lock=(WORK/LOCK_NAME).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready,reason=eval_ready()
    assert ready,reason
    env=environment()
    for rank in RANKS:
        checkpoint=ROOT/'checkpoints'/f'OrbitStiefel-paper-seed42-r{rank}'
        evidence=json.loads((WORK/f'trained-r{rank}.json').read_text())
        assert digest(checkpoint/'adapter_model.bin')==evidence['adapter_sha256']
        results=checkpoint/EVAL_FOLDER
        results.mkdir(exist_ok=True)
        for dataset in ['okvqa','ocrvqa','gqa','textvqa']:
            marker=WORK/f'evaluated-r{rank}-{dataset}.json'
            if marker.exists():continue
            while True:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                available={int(a):int(b) for a,b in (line.split(',') for line in free.splitlines())}[0]
                if available>=26000:break
                status('WAITING_MEMORY',rank=rank,dataset=dataset,free_mib=available)
                time.sleep(20)
            status('EVALUATING',rank=rank,dataset=dataset)
            with (LOG/f'r{rank}-{dataset}.log').open('a') as out:
                ACTIVE=subprocess.Popen(['bash',f'scripts/v1_5/eval/upstream/eval_{dataset}.sh',str(checkpoint),str(results),str(results/'summary.txt')],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                code=ACTIVE.wait();ACTIVE=None
            if code:raise RuntimeError(f'rank{rank} {dataset} exited {code}; training was left running')
            score=results/dataset/'llava-v1.5-7b'/f'result-{dataset}.txt'
            assert score.is_file() and score.stat().st_size
            atomic(marker,dict(result=str(score),result_sha256=digest(score),concurrent_with_training=True,gpu=0))
        atomic(results/'protocol.json',json.loads((ROOT/'configs/paper_eval_provenance.json').read_text()))
    status('COMPLETED',ranks=RANKS,evaluations=4*len(RANKS))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rank',type=int,choices=[16,32])
    args=parser.parse_args()
    if args.rank is not None:
        RANKS=[args.rank]
        STATUS_NAME=f'parallel-eval-r{args.rank}-status.json'
        LOCK_NAME=f'parallel-eval-r{args.rank}.lock'
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except Exception as exc:
        status('FAILED',error=repr(exc));raise
