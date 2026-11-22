"""Read-only experiment monitor; atomically refresh MD/HTML/JSON every 10s."""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import fcntl
import html
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('PAPER_PIPELINE_WORK', ROOT/'automation/orbit-paper-seed42'))
LOG = Path(os.environ.get('PAPER_PIPELINE_LOG', ROOT/'logs/auto-paper'))
EVAL_FOLDER = os.environ.get('PAPER_PIPELINE_EVAL_FOLDER', 'eval_results_paper_gpu0')
TAG_PREFIX = os.environ.get('PAPER_PIPELINE_TAG_PREFIX', 'OrbitStiefel-paper-seed42')
TITLE = os.environ.get('PAPER_PIPELINE_TITLE', '新方案实时进度')
EXEC_NOTE = os.environ.get('PAPER_PIPELINE_EXEC_NOTE', '训练设置：seed42；LLM rank16→32→64；connector rank128；rank16/32双卡，rank64及评测使用GPU0；3epochs；global batch16。')
GPU_LABEL = os.environ.get('PAPER_PIPELINE_GPU_LABEL', 'GPU0')
DOWNLOADS = ROOT/'assets/data/upstream/downloads'
HISTORY = defaultdict(lambda: deque(maxlen=61))
STAGES = {
    'WAITING_AUTHOR_TRAINING_JSON': '短测试完成，等待作者训练 JSON 核实',
    'RESUMING_ICONQA': '续传 IconQA', 'WAITING_ICONQA': '等待 IconQA 校验',
    'EXTRACTING_ICONQA': '解压 IconQA', 'WAITING_GPU_MEMORY': '等待显存',
    'GPU_SMOKE': '7B 短测试', 'RETRYING_FROM_CHECKPOINT': '失败后从完整断点自动恢复',
    'TRAINING': '正式训练',
    'TRAINED_WAITING_STRICT_EVAL': '训练完成，等待严格评测条件',
    'EVALUATING': '双卡评测', 'COMPLETED': '全部完成', 'FAILED': '失败，查看日志',
}
DATASETS = ['okvqa', 'ocrvqa', 'gqa', 'textvqa']
NAMES = ['OKVQA', 'OCRVQA', 'GQA', 'TextVQA']


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def atomic(path, content):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(content)
    tmp.replace(path)


def size(path):
    try:
        return path.stat().st_size
    except OSError:
        return 0


def duration(seconds):
    if seconds is None:
        return '测量中'
    if seconds < 60:
        return '<1 分钟'
    if seconds < 3600:
        return f'{seconds/60:.0f} 分钟'
    return f'{seconds/3600:.1f} 小时'


def rate(key, value, total, minimum=1):
    history = HISTORY[key]
    now = time.monotonic()
    if history and value < history[-1][1]:
        history.clear()
    history.append((now, value))
    elapsed = now-history[0][0]
    delta = value-history[0][1]
    speed = delta/elapsed if elapsed >= 20 and delta >= minimum else None
    eta = max(0, total-value)/speed if speed else None
    return speed, eta


def icon_bytes():
    total, chunk = 1851278883, 16*1024*1024
    parts = DOWNLOADS/'iconqa_data.parts'
    complete, count = 0, 0
    for start in range(0, total, chunk):
        expected = min(chunk, total-start)
        part = parts/f'{start:012d}.part'
        if size(part) == expected:
            complete += expected
            count += 1
        else:
            # The range downloader preserves a prefix in .tmp and receives
            # its next contiguous segment separately, then commits it.
            complete += min(expected, size(part.with_suffix('.tmp')) + size(part.with_suffix('.segment')))
    if (DOWNLOADS/'iconqa_data.zip.complete').exists():
        return total, 111
    return complete, count


def processes():
    live = False
    try:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                args = (proc/'cmdline').read_bytes().split(b'\0')
                if any(x.endswith(b'/auto_paper_pipeline.py') or x == b'scripts/auto_paper_pipeline.py' for x in args):
                    live = True
                    break
            except OSError:
                pass
    except OSError:
        pass
    return live


def score(path):
    try:
        match = re.search(r'Accuracy(?: on [^:]+)?:\s*([0-9.]+)%', path.read_text())
        return f'{float(match.group(1)):.2f}%' if match else '已完成，待解析'
    except OSError:
        return '结果文件缺失'


def prediction_count(folder):
    count = 0
    for index in [0, 1]:
        path = folder/f'2_{index}.jsonl'
        try:
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    count += block.count(b'\n')
        except OSError:
            pass
    return count


def snapshot():
    pipeline = read(WORK/'status.json')
    live = processes()
    stage = pipeline.get('stage', 'UNKNOWN')
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    data = []
    icon, parts = icon_bytes()
    specs = [('IconQA', 'iconqa_data.zip', 1851278883, icon),
             ('GQA 图片', 'gqa-images.zip', 21817965542, None),
             ('OCRVQA', 'ocr_vqa.zip', 8931303798, None),
             ('TextVQA 图片', 'textvqa-train-val-images.zip', 7072297970, None),
             ('COCO val2014 / OKVQA 图片', 'val2014.zip', 6645013297, None)]
    for name, filename, total, custom in specs:
        current = min(total, custom if custom is not None else size(DOWNLOADS/filename))
        done = (DOWNLOADS/(filename+'.complete')).exists()
        speed, eta = rate(filename, current, total)
        status = '下载完成' if done else '下载中' if speed else '等待/测量中'
        if filename == 'iconqa_data.zip':
            status += f'；分块 {parts}/111'
            if (WORK/'iconqa-extracted.json').exists():
                status += '；已解压'
            if (WORK/'run-manifest.json').exists():
                status += '；训练集已校验'
        data.append(dict(name=name, downloaded_gb=current/1e9, total_gb=total/1e9,
                         percent=100*current/total, status=status,
                         speed_mb_s=speed/1e6 if speed else None,
                         eta='—' if done else duration(eta)))
    provenance = read(ROOT/'configs/paper_eval_provenance.json')
    rows = []
    for rank in [16, 32, 64]:
        tag = f'{TAG_PREFIX}-r{rank}'
        checkpoint = ROOT/'checkpoints'/tag
        progress = read(ROOT/'progress'/f'{tag}.json')
        trained = read(WORK/f'trained-r{rank}.json')
        smoke = '通过' if (WORK/f'trained-r{rank}.json').exists() or (WORK/f'smoke-gpu0-r{rank}.json').exists() else '待运行'
        active = pipeline.get('rank') == rank and live
        if active and stage == 'GPU_SMOKE':
            smoke_progress = read(ROOT/'progress'/f'{tag}-smoke-gpu0.json')
            smoke = f"运行中 {smoke_progress.get('current_step', 0)}/4"
            smoke_log = LOG/f'r{rank}-smoke.log'
            if smoke_log.exists() and time.time()-smoke_log.stat().st_mtime > 300:
                smoke += '；日志超过5分钟无更新，待排查'
        if stage == 'FAILED' and pipeline.get('rank') == rank:
            smoke = '失败' if not (WORK/f'smoke-r{rank}.json').exists() else smoke
        step = int(trained.get('global_step', progress.get('current_step', 0)))
        step = min(1875, step)
        training = '完成' if trained else '训练中' if active and stage == 'TRAINING' else '待运行'
        if not trained and step:
            training = '训练中' if active and stage == 'TRAINING' else '已停止/待恢复'
        if stage == 'FAILED' and not trained and pipeline.get('rank') == rank:
            training = '失败'
        if stage == 'WAITING_AUTHOR_TRAINING_JSON' and not trained:
            training = '等待作者训练 JSON 核实'
        speed, eta = rate(f'train-r{rank}', step, 1875, minimum=10)
        evaluation = []
        for dataset in DATASETS:
            marker = WORK/f'evaluated-r{rank}-{dataset}.json'
            folder = checkpoint/EVAL_FOLDER/dataset/'llava-v1.5-7b'
            if marker.exists():
                value = score(folder/f'result-{dataset}.txt')
            elif (read(WORK/f'parallel-eval-r{rank}-status.json').get('stage') == 'PAUSED_QUEUED' and
                  read(WORK/f'parallel-eval-r{rank}-status.json').get('dataset') == dataset):
                value = f'排队（保留{prediction_count(folder)}条预测）'
            elif ((active and stage == 'EVALUATING' and pipeline.get('dataset') == dataset) or
                  (read(WORK/'parallel-eval-status.json').get('stage') == 'EVALUATING' and
                   read(WORK/'parallel-eval-status.json').get('rank') == rank and
                   read(WORK/'parallel-eval-status.json').get('dataset') == dataset) or
                  (read(WORK/f'parallel-eval-r{rank}-status.json').get('stage') == 'EVALUATING' and
                   read(WORK/f'parallel-eval-r{rank}-status.json').get('dataset') == dataset)):
                expected = dict(okvqa=5046, gqa=12578, textvqa=5000,
                                ocrvqa=provenance.get('ocrvqa', {}).get('count')).get(dataset)
                value = f'推理 {prediction_count(folder)}/{expected or "待核实"}'
            else:
                value = '待评测' if trained else '等待训练'
            evaluation.append(value)
        rows.append(dict(rank=rank, seed=42, smoke=smoke, training=training,
                         steps=step, total_steps=1875, percent=100*step/1875,
                         epoch=3 if trained else round(3*step/1875, 3),
                         eta='—' if trained else duration(eta) if training=='训练中' else '待开始',
                         evaluations=evaluation,
                         log=str(LOG/f'r{rank}-{"smoke" if stage == "GPU_SMOKE" else "train"}.log')))
    reasons = []
    if not read(ROOT/'configs/paper_training_provenance.json').get('verified'):
        reasons.append('论文训练抽样/作者转换模板尚未校验。')
    if not provenance.get('ocrvqa', {}).get('verified'):
        reasons.append('OCRVQA 原论文抽样文件及来源未核实；本地子集不会自动替代。')
    gqa = ROOT/'assets/data/eval/gqa/data'
    if not (gqa/'testdev_balanced_questions.json').exists():
        reasons.append('GQA 官方 testdev-balanced 答案文件缺失。')
    return dict(updated_at=now, refresh_seconds=10, pipeline_alive=live,
                stage=stage, stage_label=STAGES.get(stage, stage),
                stage_updated_at=pipeline.get('updated_at', ''),
                reason=pipeline.get('reason') or pipeline.get('error') or '',
                downloads=data, experiments=rows, strict_eval_pending=reasons)


def render(state):
    training_columns = ['Rank', 'Seed', '运行短测', '训练状态', '训练步数', '进度', 'Epoch', '训练剩余估算']+NAMES
    training_rows = [[str(r['rank']), '42', r['smoke'], r['training'],
                      f"{r['steps']}/1875", f"{r['percent']:.1f}%", f"{r['epoch']:.3f}", r['eta']]+r['evaluations']
                     for r in state['experiments']]
    data_columns = ['数据', '已下载 / 总量（GB）', '进度', '状态', '速度（MB/s）', '下载剩余估算']
    data_rows = [[d['name'], f"{d['downloaded_gb']:.2f} / {d['total_gb']:.2f}",
                  f"{d['percent']:.1f}%", d['status'],
                  f"{d['speed_mb_s']:.2f}" if d['speed_mb_s'] is not None else '测量中', d['eta']]
                 for d in state['downloads']]
    health = '运行中' if state['pipeline_alive'] else '未运行（检查主日志）'
    summary = f"自动队列：{health}；当前阶段：{state['stage_label']}。"
    table = lambda columns, rows: '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(row)+' |' for row in rows])
    notes = ['OCRVQA：官方test随机2000条、seed42固定抽样，图片答案已核验；原作者具体抽样ID未取得，属于协议复现。',
             '每10秒自动写入；HTML页面自动刷新，Markdown预览是否自动刷新取决于编辑器。',
             EXEC_NOTE,
             '剩余时间仅按实测速度估算。训练开始并推进至少10步后才显示估算；不预估尚未开始的训练或完整队列时间。',
             '下载完成不代表解压、数据校验或严格论文评测协议已通过。']
    md = [f'# {TITLE}', '', f"更新时间：{state['updated_at']}（10秒刷新）", '', summary, '',
          table(training_columns, training_rows), '', '## 数据下载', '', table(data_columns, data_rows), '',
          '## 严格评测等待项', '']
    md += ['- '+reason for reason in state['strict_eval_pending']] or ['- 等待项已解除，实际校验以自动队列为准。']
    if state['reason']:
        md += ['', '队列提示：'+str(state['reason'])]
    md += ['', '## 说明与日志', '']+['- '+note for note in notes]
    md += ['', f"[主日志]({LOG/'pipeline.log'})"]
    for row in state['experiments']:
        md += [f"[rank{row['rank']}训练日志]({row['log']})"]
    atomic(WORK/'live.md', '\n'.join(md)+'\n')
    def htable(columns, rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="10"><title>'+html.escape(TITLE)+'</title><style>body{font:15px system-ui,sans-serif;background:#f5f7fb;color:#172338;padding:24px;max-width:1600px;margin:auto}h1{font-size:25px}.scroll{overflow:auto;background:white;border:1px solid #dce3ee;border-radius:10px;margin:16px 0}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:13px 16px;text-align:left;border-bottom:1px solid #edf0f5}th{background:#edf3fa}small{color:#56677e}.note{background:#fff4dd;padding:16px;border-radius:10px;line-height:1.8}li{margin:8px 0}</style><h1>'+html.escape(TITLE)+'</h1>'
    page += '<small>'+html.escape(state['updated_at'])+' · 每10秒自动刷新</small><p>'+html.escape(summary)+'</p>'
    page += htable(training_columns, training_rows)+'<h2>数据下载</h2>'+htable(data_columns, data_rows)
    page += '<div class="note"><b>严格评测等待项</b><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in state['strict_eval_pending'])+'</ul>'+html.escape(str(state['reason']))+'</div>'
    page += '<ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in notes)+'</ul>'
    links = [('主日志', LOG/'pipeline.log')]+[(f"rank{r['rank']}训练日志", Path(r['log'])) for r in state['experiments']]
    page += '<p>'+ ' · '.join('<a href="'+html.escape(os.path.relpath(p, WORK))+'">'+name+'</a>' for name, p in links)+'</p></html>'
    atomic(WORK/'live.html', page)
    atomic(WORK/'live.json', json.dumps(state, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    options = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK/'monitor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            render(snapshot())
        except Exception as error:
            print(datetime.now(timezone.utc).isoformat(), repr(error), flush=True)
            if options.once:
                raise
        if options.once:
            break
        time.sleep(10)
