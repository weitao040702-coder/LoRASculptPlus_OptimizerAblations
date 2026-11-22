"""Reconstruct paper-protocol IconQA choose-text training examples.

Paper §4.1 specifies random 10k examples, not identical author sample IDs.
Choice/answer formatting follows the author's cited Mantis prepare.py at
912942133110beaa10dffe0b0ddac0eb851ec61a. The images/from schema is adapted
to LLaVA's single-image image/from schema; the textual values are preserved.
"""
import json
import random
from pathlib import Path


def build_records(root, split_map, problems, sample_count=10000, seed=42):
    root = Path(root)
    ids = random.Random(seed).sample(split_map['choose_txt_train'], sample_count)
    placement = random.Random(seed)
    records = []
    for pid in ids:
        item = problems[str(pid)]
        answer = int(item['answer'])
        assert 0 <= answer < len(item['choices']) <= 26
        choices = '\n'.join(f'{chr(65+i)}. {text}' for i, text in enumerate(item['choices']))
        question = item['question']
        question = '<image>'+question if placement.random() < .5 else question+'<image>'
        image = f'iconqa/train/choose_txt/{pid}/image.png'
        if not (root/image).is_file():
            raise FileNotFoundError(root/image)
        records.append(dict(question_id=str(pid), image=image,
            conversations=[dict(from_='human', value=question+'\n'+choices),
                           dict(from_='gpt', value='Answer: '+chr(65+answer))]))
        for turn in records[-1]['conversations']:
            turn['from'] = turn.pop('from_')
    assert len({x['question_id'] for x in records}) == sample_count
    return records
