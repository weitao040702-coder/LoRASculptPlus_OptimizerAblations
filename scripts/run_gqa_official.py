"""Run the published GQA evaluator without nonstandard CLI arguments."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--evaluator', type=Path, required=True)
parser.add_argument('--predictions', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--summary-output-dir', default='None')
args = parser.parse_args()
root = args.data_root.resolve()
result = subprocess.run([sys.executable, str(args.evaluator.resolve()), '--tier', 'testdev_balanced',
                         '--scenes', str(root/'testdev_balanced_sceneGraphs.json'),
                         '--questions', str(root/'testdev_balanced_questions.json'),
                         '--choices', str(root/'testdev_balanced_choices.json'),
                         '--predictions', str(args.predictions.resolve())], cwd=root,
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout, end='')
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir/'official-evaluator.log').write_text(result.stdout)
if result.returncode:
    raise SystemExit(result.returncode)
match = re.search(r'^Accuracy:\s*([0-9.]+)%', result.stdout, re.MULTILINE)
if not match:
    raise RuntimeError('Official evaluator returned no Accuracy score')
line = f'Accuracy: {float(match.group(1)):.2f}%\n'
(args.output_dir/'result-gqa.txt').write_text(line)
if args.summary_output_dir != 'None':
    with Path(args.summary_output_dir).open('a') as handle:
        handle.write('GQA official evaluator '+line)
