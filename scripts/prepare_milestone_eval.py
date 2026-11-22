#!/usr/bin/env python3
"""Prepare OKVQA and the recorded seed-42 OCRVQA evaluation instances."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from prepare_upstream_eval import prepare_okvqa

SOURCE_SHA = 'c9d2bb4c67462e2649be5099a3b790c95ad073fe46243310b79a1d4c8bee75ed'
SUBSET_SHA = '9a45d9928cbe35d7a89d0dcd099be67e5c244465ace4d8335ad2be6a057496e9'


def main():
    """Check input metadata and reconstruct exactly the recorded question subset."""
    p = argparse.ArgumentParser()
    p.add_argument('--eval-root', type=Path, required=True)
    p.add_argument('--ocr-metadata', type=Path, required=True)
    args = p.parse_args()
    payload = args.ocr_metadata.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError('OCRVQA metadata SHA-256 mismatch')
    prepare_okvqa(args.eval_root)
    data = json.loads(payload)
    candidates = []
    for image_id in sorted(data):
        book = data[image_id]
        if book['split'] != 3:
            continue
        for index, (question, answer) in enumerate(zip(book['questions'], book['answers'])):
            candidates.append(dict(question_id=f'{image_id}_{index}',
                image=image_id + Path(book['imageURL']).suffix.lower(),
                text=question+'\nAnswer the question using a single word or phrase.',
                answer=answer, category='default', source_image_id=image_id,
                source_question_index=index, source_split=3))
    if len(candidates) != 100429:
        raise ValueError(f'Unexpected OCRVQA population: {len(candidates)}')
    rows = random.Random(42).sample(candidates, 2000)
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
    if hashlib.sha256(content).hexdigest() != SUBSET_SHA:
        raise ValueError('OCRVQA subset SHA-256 mismatch')
    folder = args.eval_root / 'ocrvqa'
    images = folder / 'sampled_images'
    images.mkdir(parents=True, exist_ok=True)
    for name in sorted({r['image'] for r in rows}):
        source = (folder / 'raw' / 'images' / name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = images / name
        if target.exists():
            if not target.samefile(source):
                raise FileExistsError(f'Existing image differs: {target}')
        else:
            target.symlink_to(source)
    output = folder / 'sampled_ocrvqa_test.jsonl'
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f'Refusing to overwrite differing file: {output}')
    output.write_bytes(content)
    print(f'OCRVQA: {len(rows)} samples; sha256={SUBSET_SHA}; output={output}')


if __name__ == '__main__':
    main()
