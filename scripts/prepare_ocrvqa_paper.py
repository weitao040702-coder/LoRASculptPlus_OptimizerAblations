"""Reconstruct the author's random-2000 OCRVQA protocol from official test QA.

The author subset IDs are unavailable; this is a recorded local sample, not
an assertion that the published subset or its exact score is reproduced.
"""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'assets/data/upstream/downloads/ocrvqa-official-dataset.json'
SOURCE_SHA = 'c9d2bb4c67462e2649be5099a3b790c95ad073fe46243310b79a1d4c8bee75ed'
REVISION = 'f6f6a8246fa846768bef63bbe7606e0ad1810162'
SEED, COUNT = 42, 2000

def main():
    payload = SOURCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA, 'Metadata checksum differs'
    data = json.loads(payload)
    candidates = []
    for image_id in sorted(data):
        book = data[image_id]
        if book['split'] != 3:
            continue
        assert len(book['questions']) == len(book['answers'])
        for index, (question, answer) in enumerate(zip(book['questions'], book['answers'])):
            assert isinstance(question, str) and question and isinstance(answer, str) and answer
            candidates.append(dict(question_id=f'{image_id}_{index}', image=image_id + Path(book['imageURL']).suffix.lower(),
                text=question+'\nAnswer the question using a single word or phrase.',
                answer=answer, category='default', source_image_id=image_id,
                source_question_index=index, source_split=3))
    assert len(candidates) == 100429
    rows = random.Random(SEED).sample(candidates, COUNT)
    raw_images = ROOT / 'assets/data/eval/ocrvqa/raw/images'
    # Do not silently discard missing samples or resample to hide missing images.
    from PIL import Image
    for name in sorted({r['image'] for r in rows}):
        with Image.open(raw_images / name) as im:
            im.verify()
    dest = ROOT / 'assets/data/eval/ocrvqa'
    images = dest / 'sampled_images'
    images.mkdir(parents=True, exist_ok=True)
    for name in {r['image'] for r in rows}:
        path = images / name
        if path.is_symlink():
            assert path.resolve() == (raw_images / name).resolve()
        elif path.exists():
            assert path.read_bytes() == (raw_images / name).read_bytes()
        else:
            path.symlink_to(Path('../raw/images') / name)
    output = dest / 'sampled_ocrvqa_test.jsonl'
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
    if output.exists():
        assert output.read_bytes() == content, 'Existing subset differs; preserve before replacing'
    else:
        output.write_bytes(content)
    assert len({r['question_id'] for r in rows}) == COUNT
    evidence = dict(verified=True, verification_scope='paper_protocol_reconstruction',
        author_subset_identical=False,
        source=f'https://huggingface.co/datasets/nick121381/ocr_vqa_json/blob/{REVISION}/dataset.json',
        protocol_source='https://github.com/LiangJian24/LoRASculpt/issues/2',
        metadata_sha256=SOURCE_SHA, sha256=hashlib.sha256(content).hexdigest(),
        count=COUNT, image_count=len({r['image'] for r in rows}), split='official test (split=3)',
        population_questions=len(candidates), sampling_unit='question-answer pair',
        sampling_seed=SEED, sampling_order='sorted image_id, then original question index',
        prompt='Original question + newline + Answer the question using a single word or phrase.',
        metric='Case-insensitive, whitespace-stripped exact match; LLaVA-HR OCRVQAAccuracyEvaluator',
        reason='Author confirms random 2000 evaluation instances. Reconstructed from official test QA; original author subset IDs unavailable. Published scores are not an exact same-subset comparison.')
    evidence_path = ROOT / 'configs/paper_eval_provenance.json'
    provenance = json.loads(evidence_path.read_text())
    provenance['ocrvqa'] = evidence
    evidence_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+'\n')
    record = ROOT / 'automation/orbit-paper-seed42/ocrvqa-metadata-provenance.json'
    record.write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(evidence, indent=2))

if __name__ == '__main__':
    main()
