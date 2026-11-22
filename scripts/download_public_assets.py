"""Download public models and raw datasets with resumable curl transfers."""
import concurrent.futures as cf
import json, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs/public-assets'; LOG.mkdir(parents=True,exist_ok=True)
BASE='https://hf-mirror.com'

def fetch(url,out):
 out=ROOT/out;out.parent.mkdir(parents=True,exist_ok=True)
 marker=out.with_name(out.name+'.complete')
 if marker.exists() and out.exists():return
 print('DOWNLOADING',out.relative_to(ROOT),flush=True)
 with (LOG/(out.name+'.log')).open('a') as log:
  subprocess.run(['curl','--fail','-L','--retry','8','--retry-all-errors','--retry-delay','5','--connect-timeout','30','--speed-time','120','--speed-limit','1024','-C','-','-o',str(out),url],stdout=log,stderr=log,check=True)
 marker.write_text(url+'\n')
 print('DOWNLOADED',out.relative_to(ROOT),out.stat().st_size,flush=True)

def model(repo,folder):
 result=subprocess.run(['curl','-fsSL','--retry','3','--max-time','60',BASE+'/api/models/'+repo],capture_output=True,check=True)
 info=json.loads(result.stdout)
 (LOG/(folder+'-revision.json')).write_text(json.dumps(info,indent=2))
 files=[x['rfilename'] for x in info['siblings'] if x['rfilename'].endswith(('.json','.model','.bin','.safetensors','.txt')) and not x['rfilename'].startswith(('onnx/','onnx','tf_model','flax_model'))]
 # Use PyTorch weights only when a repository offers duplicate formats.
 if any(f.endswith('.bin') for f in files):files=[f for f in files if not f.endswith('.safetensors')]
 with cf.ThreadPoolExecutor(max_workers=3) as pool:
  list(pool.map(lambda f:fetch(f'{BASE}/{repo}/resolve/{info["sha"]}/{f}',f'assets/models/{folder}/{f}'),files))

D='assets/data/upstream/downloads/'
JOBS=[
('https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip','iconqa_data.zip'),
('https://hf-mirror.com/datasets/zrchen03/llava_eval/resolve/main/gqa/images.zip','gqa-images.zip'),
('https://hf-mirror.com/datasets/zrchen03/llava_eval/resolve/main/gqa/eval.zip','gqa-eval.zip'),
('https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json','TextVQA_0.5.1_val.json'),
('https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip','textvqa-train-val-images.zip'),
('https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip','mscoco_val2014_annotations.json.zip'),
('https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip','OpenEnded_mscoco_val2014_questions.json.zip'),
('https://hf-mirror.com/datasets/zrchen03/llava_eval/resolve/main/pope/val2014.zip','val2014.zip'),
(BASE+'/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip','ocr_vqa.zip'),
('https://hf-mirror.com/datasets/Xiaodong/LLaVA-1.5-Eval/resolve/main/eval.zip','llava-eval.zip')]

def data(url,name):
 if name=='iconqa_data.zip' and not (ROOT/D/(name+'.complete')).exists():
  subprocess.run(['python3','-u',str(ROOT/'scripts/download_iconqa_ranges.py')],check=True)
 else:
  fetch(url,D+name)
 archive=ROOT/D/name
 if name.endswith('.zip'):
  subprocess.run(['unzip','-tq',str(archive)],stdout=subprocess.DEVNULL,check=True)
  targets={'iconqa_data.zip':'assets/data/iconqa/raw/data','gqa-images.zip':'assets/data/eval/gqa/data','gqa-eval.zip':'assets/data/eval/gqa/data','textvqa-train-val-images.zip':'assets/data/eval/textvqa','mscoco_val2014_annotations.json.zip':'assets/data/eval/okvqa','OpenEnded_mscoco_val2014_questions.json.zip':'assets/data/eval/okvqa','val2014.zip':'assets/data/eval/okvqa','ocr_vqa.zip':'assets/data/eval/ocrvqa/raw','llava-eval.zip':'assets/data/eval'}
  target=ROOT/targets[name];target.mkdir(parents=True,exist_ok=True)
  subprocess.run(['unzip','-q','-o',str(archive),'-d',str(target)],stdin=subprocess.DEVNULL,check=True)
 else:
  import shutil
  target=ROOT/'assets/data/eval/textvqa';target.mkdir(parents=True,exist_ok=True);shutil.copy2(archive,target/name)
 print('READY',name,flush=True)

def main():
 tasks=[];errors=[]
 with cf.ThreadPoolExecutor(max_workers=6) as pool:
  tasks += [pool.submit(model,*m) for m in [('liuhaotian/llava-v1.5-7b','llava-v1.5-7b'),('openai/clip-vit-large-patch14-336','clip-vit-large-patch14-336')]]
  tasks += [pool.submit(data,*j) for j in JOBS]
  for task in cf.as_completed(tasks):
   try:task.result()
   except Exception as e:errors.append(str(e));print('FAILED',str(e),flush=True)
 (LOG/'status.json').write_text(json.dumps({'finished':True,'errors':errors},indent=2))
 print('ALL_DOWNLOAD_TASKS_FINISHED',len(errors),'errors',flush=True)
 subprocess.run(['python3',str(ROOT/'scripts/verify_downloaded_public_assets.py')],check=False)
if __name__=='__main__':main()
