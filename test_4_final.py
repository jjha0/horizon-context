#!/usr/bin/env python3
"""Frozen Test comparison: Stage3 plain vs selected Stage4 Replay with identical recovery.
No model training or model selection. Uses the previously prepared Test JSONL.
Examples are processed once per model and cached so interrupted runs can resume.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''):
            h.update(x)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_rows(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score(records, glossary):
    from sacrebleu.metrics import CHRF, BLEU
    from prompts import terms, follows
    refs = [r['reference_ko'] for r in records]
    preds = [r['prediction_ko'] for r in records]
    context = [r for r in records if r['has_context']]
    num = hits = 0
    for r in records:
        for term in terms(r['current_en'], glossary):
            num += 1
            hits += int(follows(r['prediction_ko'], term))
    out = {
        'n': len(records), 'n_context': len(context),
        'chrf': CHRF().corpus_score(preds, [refs]).score,
        'chrf_context': CHRF().corpus_score([r['prediction_ko'] for r in context],
                                           [[r['reference_ko'] for r in context]]).score if context else None,
        'empty': sum(not r['prediction_ko'].strip() for r in records),
        'hit_cap': sum(bool(r['hit_cap']) for r in records),
        'still_abnormal': sum(bool(r['abnormal_reasons']) for r in records),
        'retry_used': sum(bool(r['retry_used']) for r in records),
        'term_n': num, 'term_hit': hits,
    }
    try:
        bleu = BLEU(tokenize='ko-mecab')
        out['bleu_ko_mecab'] = bleu.corpus_score(preds, [refs]).score
    except Exception as e:
        out['bleu_ko_mecab'] = None
        out['bleu_error'] = str(e)
    return out


def evaluate(name, mode, adapter, rows, release, output):
    from infer_stage4 import Stage4Translator
    from prompts import prompt
    import torch
    path = output / (name + '.jsonl')
    saved = {}
    if path.exists():
        for r in read_rows(path):
            if r['id'] in saved:
                raise RuntimeError('Duplicate cached Test record: ' + r['id'])
            saved[r['id']] = r
        if set(saved) - {r['id'] for r in rows}:
            raise RuntimeError('Test cache includes unexpected sample IDs')
    remaining = len(rows) - len(saved)
    if remaining:
        translator = Stage4Translator(adapter_dir=adapter,
                                      glossary_path=release / 'strict_glossary.json',
                                      max_input_tokens=1536, max_new_tokens=512)
        if mode == 'plain':
            translator.build_prompt = lambda current_en, prev_en='': prompt(
                {'current_en': current_en, 'prev_en': ''}, 'plain', translator.glossary)
        print(f'Running {name}: {len(saved)}/{len(rows)} done', flush=True)
        with path.open('a', encoding='utf-8') as f:
            for i, row in enumerate(rows, 1):
                if row['id'] in saved:
                    continue
                result = translator.translate(row['current_en'], row['prev_en'] if mode == 'context' else '')
                record = {
                    'id': row['id'], 'paper_id': row['paper_id'],
                    'current_en': row['current_en'], 'prev_en': row['prev_en'],
                    'reference_ko': row['target_ko'], 'prediction_ko': result['translation'],
                    'has_context': bool(row['prev_en'].strip()),
                    'hit_cap': result['hit_cap'], 'retry_used': result['retry_used'],
                    'abnormal_reasons': result['abnormal_reasons'], 'model': name, 'mode': mode,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                saved[record['id']] = record
                if len(saved) % 10 == 0 or len(saved) == len(rows):
                    print(f'{name}: {len(saved)}/{len(rows)}', flush=True)
        del translator
        gc.collect()
        torch.cuda.empty_cache()
    if len(saved) != len(rows):
        raise RuntimeError('Incomplete model predictions; do not score partial Test results')
    return [saved[r['id']] for r in rows]


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--release', default='/workspace/stage4_handoff_20261104_024558/release_content')
    a.add_argument('--stage3', default='/workspace/terminology_model/adapter')
    a.add_argument('--replay', default='/workspace/horizon_context_lora/runs/replay_continue_01/best_adapter')
    a.add_argument('--test', default='/workspace/horizon_context/data/test.jsonl')
    a.add_argument('--output', default='/workspace/horizon_context_lora/runs/final_test_replay')
    a.add_argument('--check-only', action='store_true')
    args = a.parse_args()
    release = Path(args.release).resolve()
    stage3, replay = Path(args.stage3).resolve(), Path(args.replay).resolve()
    test = Path(args.test).resolve()
    out = Path(args.output).resolve()
    for path in (release / 'infer_stage4.py', release / 'prompts.py', release / 'strict_glossary.json',
                 test, stage3 / 'adapter_config.json', stage3 / 'adapter_model.safetensors',
                 replay / 'adapter_config.json', replay / 'adapter_model.safetensors'):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_rows(test)
    ids = set()
    for r in rows:
        if r.get('split') != 'test' or r.get('alignment_type') != '1:1':
            raise ValueError('Expected unchanged 1:1 Test split; no filtering or replacement')
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ('id', 'paper_id', 'current_en', 'target_ko')):
            raise ValueError('Missing required Test sample fields')
        if not isinstance(r.get('prev_en'), str) or r['id'] in ids:
            raise ValueError('Missing context or duplicate Test ID')
        ids.add(r['id'])
    if not rows:
        raise ValueError('No Test examples')
    for mod in (release/'infer_stage4.py', release/'prompts.py'):
        if not mod.is_file(): raise FileNotFoundError(mod)
    manifest = {
        'test_file_sha256': digest(test), 'n_test': len(rows),
        'stage3_adapter_config_sha256': digest(stage3 / 'adapter_config.json'),
        'stage3_adapter_weights_sha256': digest(stage3 / 'adapter_model.safetensors'),
        'replay_adapter_config_sha256': digest(replay / 'adapter_config.json'),
        'replay_adapter_weights_sha256': digest(replay / 'adapter_model.safetensors'),
        'infer_sha256': digest(release / 'infer_stage4.py'),
        'prompt_sha256': digest(release / 'prompts.py'),
        'glossary_sha256': digest(release / 'strict_glossary.json'),
        'max_input_tokens': 1536, 'max_new_tokens': 512,
        'baseline_mode': 'plain', 'replay_mode': 'context',
        'output_recovery': 'conditional_repeat_recovery_v1',
        'test_used_for_training_or_selection': False,
    }
    print(f'CHECK PASSED | Test {len(rows)} | stage3 and replay files present', flush=True)
    if args.check_only: return
    if out.exists() and (out / 'final_report.json').exists():
        raise RuntimeError('Test already completed. Do not rerun or use it for more model selection.')
    out.mkdir(parents=True, exist_ok=True)
    meta = out / 'manifest.json'
    if meta.exists():
        if json.loads(meta.read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('Data/model/config changed; refuse to mix predictions with previous Test.')
    elif any(out.iterdir()):
        raise RuntimeError('Nonempty output without manifest; choose a clean output folder.')
    else:
        save_json(meta, manifest)
    sys.path.insert(0, str(release))
    glossary = json.loads((release / 'strict_glossary.json').read_text(encoding='utf-8'))
    b = evaluate('stage3_plain', 'plain', stage3, rows, release, out)
    c = evaluate('replay_context', 'context', replay, rows, release, out)
    bm, cm = score(b, glossary), score(c, glossary)
    report = {'status': 'COMPLETE', 'n_test': len(rows), 'stage3': bm, 'replay': cm,
              'delta': {k: (cm[k] - bm[k] if cm[k] is not None and bm[k] is not None else None)
                        for k in ('chrf', 'bleu_ko_mecab', 'chrf_context')},
              'model_selection_changed': False, 'training_performed': False,
              'note': 'Same fixed 256-row sample from the originally prepared Test split; Stage3 plain versus Replay context; conditional retry on abnormal outputs for both.'}
    save_json(out / 'final_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('TEST COMPLETE:', out / 'final_report.json', flush=True)


if __name__ == '__main__':
    main()
