"""Prepare public assets and check original protocol fingerprints, without training."""
import hashlib,importlib.util,json,random,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'assets/data/eval'
# LLaVA's archive sometimes includes its own eval/ directory.
for name in ['gqa','textvqa']:
 for filename in (['llava_gqa_testdev_balanced.jsonl'] if name=='gqa' else ['llava_textvqa_val_v051_ocr.jsonl']):
  target=E/name/filename
  if not target.exists():
   sources=list(E.rglob(filename))
   if sources:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(sources[0],target)
I=ROOT/'assets/data/iconqa'; expected=I/'raw/data/iconqa_data'
sources=list((I/'raw').rglob('pid_splits.json'))
if sources:
 source=sources[0].parent
 spec=importlib.util.spec_from_file_location('iconprep',ROOT/'scripts/prepare_iconqa_lora.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 records=mod.build_records(source,json.loads((source/'pid_splits.json').read_text()),json.loads((source/'problems.json').read_text()),'train',check_images=True)
 records=random.Random(0).sample(records,10000)
 (I/'iconqa_txt-train-10000k-seed0.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 print('ICONQA_SOURCE_ROOT',source,flush=True)
subprocess.run(['python3',str(ROOT/'scripts/prepare_upstream_eval.py'),'--eval-root',str(E)],check=False)
checks={
'assets/data/iconqa/iconqa_txt-train-10000k-seed0.json':'89ba123884d07eef2c977d4c4b27a053e4f2f0b91c1a69c15c355919f2599fe1',
'assets/data/eval/okvqa/okvqa_val.jsonl':'daa1e91a0934a0f7ef4f6aa5790c93a26b8868e4eead987e1012abed9def53e7',
'assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl':'6d6c961636a070fe59ebb1c2daf8849c1c7d7ea8c31d51e3c8c10f8b5af600a9',
'assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl':'515aab3142889269480c71212d3b5b8118704d8da6637d767a0cb5bdf80a3d35',
'assets/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl':'9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3'}
report={}
for path,wanted in checks.items():
 p=ROOT/path;actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
 report[path]={'expected_sha256':wanted,'actual_sha256':actual,'matches':actual==wanted};print(path,report[path],flush=True)
validation=subprocess.run(['python3',str(ROOT/'scripts/verify_upstream_eval.py'),'--eval-root',str(E)])
report['image_annotation_check_exit_code']=validation.returncode
(ROOT/'logs/public-assets/validation.json').write_text(json.dumps(report,indent=2))
