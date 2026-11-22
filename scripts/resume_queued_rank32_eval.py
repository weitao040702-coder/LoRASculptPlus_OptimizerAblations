"""Resume the stopped rank32 evaluator after rank16 evaluation finishes."""
import json
import os
from pathlib import Path
import signal
import time
from auto_paper_pipeline import WORK, atomic

WORKER_PID=526114
EVAL_GROUP=526143

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

while True:
    complete=all((WORK/f'evaluated-r16-{d}.json').is_file() for d in ['okvqa','ocrvqa','gqa','textvqa'])
    if complete or not alive(522553):
        if alive(EVAL_GROUP):os.killpg(EVAL_GROUP,signal.SIGCONT)
        if alive(WORKER_PID):os.kill(WORKER_PID,signal.SIGCONT)
        path=WORK/'parallel-eval-r32-status.json'
        state=json.loads(path.read_text());state.update(stage='EVALUATING',updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),resumed_after_rank16=complete)
        atomic(path,state)
        atomic(WORK/'rank32-eval-queue.json',dict(stage='RESUMED',rank16_completed=complete,worker_pid=WORKER_PID))
        break
    time.sleep(10)
