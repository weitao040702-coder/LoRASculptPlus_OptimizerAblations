"""Fetch the unmodified official IconQA archive with verified byte ranges."""
from pathlib import Path
import concurrent.futures as cf,subprocess,shutil,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/data/upstream/downloads/iconqa_data.zip'
PARTS=OUT.parent/'iconqa_data.parts';PARTS.mkdir(parents=True,exist_ok=True)
URL='https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip'
SIZE=1851278883;CHUNK=16*1024*1024

def fetch(start):
 end=min(start+CHUNK,SIZE)-1; expected=end-start+1
 part=PARTS/f'{start:012d}.part'
 if part.exists() and part.stat().st_size==expected:return part
 tmp=part.with_suffix('.tmp');headers=part.with_suffix('.headers')
 segment=part.with_suffix('.segment');segment_headers=part.with_suffix('.segment.headers')
 h=headers.read_text(errors='replace').lower() if headers.exists() else ''
 if tmp.exists() and (tmp.stat().st_size>expected or f'content-range: bytes {start}-{end}/{SIZE}' not in h):
  tmp.unlink()
 for attempt in range(12):
  offset=tmp.stat().st_size if tmp.exists() else 0
  if offset==expected:
   tmp.replace(part);return part
  first=start+offset
  subprocess.run(['curl','-fsSL','--connect-timeout','20','--max-time','600','--range',f'{first}-{end}','-D',str(segment_headers),'-o',str(segment),URL],capture_output=True)
  h=segment_headers.read_text(errors='replace').lower() if segment_headers.exists() else ''
  count=segment.stat().st_size if segment.exists() else 0
  # Keep a valid partial response even when curl hits its time limit.
  if count and count<=expected-offset and f'content-range: bytes {first}-{end}/{SIZE}' in h:
   with tmp.open('ab') as dst, segment.open('rb') as src:shutil.copyfileobj(src,dst)
   headers.write_text(f'content-range: bytes {start}-{end}/{SIZE}\n')
   segment.unlink()
  if tmp.exists() and tmp.stat().st_size==expected:
   tmp.replace(part);return part
  time.sleep(min(2*(attempt+1),10))
 raise RuntimeError(f'Range failed: {start}-{end}')

def main():
 starts=list(range(0,SIZE,CHUNK));beg=time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=16) as pool:
  futures=[pool.submit(fetch,s) for s in starts]
  for i,f in enumerate(cf.as_completed(futures),1):
   f.result();print(f'ICONQA_PART {i}/{len(starts)} elapsed={time.monotonic()-beg:.0f}s',flush=True)
 assembled=OUT.with_suffix('.assembled')
 with assembled.open('wb') as dst:
  for start in starts:
   with (PARTS/f'{start:012d}.part').open('rb') as src:shutil.copyfileobj(src,dst,16*1024*1024)
 if assembled.stat().st_size!=SIZE:raise RuntimeError('Wrong archive size')
 subprocess.run(['unzip','-tq',str(assembled)],stdout=subprocess.DEVNULL,check=True)
 assembled.replace(OUT);OUT.with_name(OUT.name+'.complete').write_text(URL+'\n')
 print('ICONQA_ARCHIVE_VERIFIED',SIZE,flush=True)
if __name__=='__main__':main()
