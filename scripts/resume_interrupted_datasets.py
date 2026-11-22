"""Recover quota-interrupted datasets while existing transfers continue."""
from pathlib import Path
import concurrent.futures as cf
import subprocess, time, json, os
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs/public-assets'
def extract(name,target):
    subprocess.run(['unzip','-tq',str(ROOT/'assets/data/upstream/downloads'/name)],stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['unzip','-q','-o',str(ROOT/'assets/data/upstream/downloads'/name),'-d',str(ROOT/target)],stdin=subprocess.DEVNULL,check=True)
    print('RECOVERED',name,flush=True)
def icon():
    subprocess.run(['python3','-u',str(ROOT/'scripts/download_iconqa_ranges.py')],check=True)
    extract('iconqa_data.zip','assets/data/iconqa/raw/data')
def main():
    errors=[]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(icon),pool.submit(extract,'val2014.zip','assets/data/eval/okvqa')]
        for f in cf.as_completed(jobs):
            try:f.result()
            except Exception as e:errors.append(str(e));print('RECOVERY_FAILED',e,flush=True)
    while Path('/proc/3676132').exists():time.sleep(20)
    result=subprocess.run(['python3','-u',str(ROOT/'scripts/verify_downloaded_public_assets.py')])
    (LOG/'recovery-status.json').write_text(json.dumps({'finished':True,'errors':errors,'verification_exit_code':result.returncode},indent=2))
if __name__=='__main__':main()
