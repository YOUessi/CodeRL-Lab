#!/usr/bin/env python3
"""Read-only stage-one presentation and evidence verification.

Default demo needs only Python 3.10+ stdlib. No model loading, Docker, network,
subprocesses, writes, or execution of generated code. Bootstrap optionally uses
NumPy. An owner artifact audit reads hashes and outcomes, never hidden test text.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

MODELS = ('sft', 'narrow_43', 'broad_43', 'narrow_44', 'broad_44')
FIELDS = ('correct', 'first', 'coverage', 'name_error', 'resource')
HEADER = ['task_id'] + [f'{n}_{k}' for n in MODELS for k in FIELDS]
TASKS = 96
DRAWS = 4
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('path escapes the selected root')
    return path


def read_json(path, expected_sha=None):
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError('SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text(encoding='utf8'))


def manifest(repo):
    m = read_json(contained(repo, 'releases/stage1/manifest.json'))
    if (m.get('schema') != 1 or m.get('task_count') != TASKS or m.get('draws_per_task') != DRAWS
            or m.get('models') != list(MODELS) or m.get('fields') != list(FIELDS)
            or m.get('private_assets_included') is not False):
        raise ValueError('unsupported or inconsistent release manifest')
    return m


def validate_rows(records):
    if len(records) != TASKS:
        raise ValueError('exactly 96 task rows required')
    found = {}
    for raw in records:
        if set(raw) != set(HEADER):
            raise ValueError('unexpected CSV columns; no raw assets permitted')
        q = raw['task_id']
        if not re.fullmatch(r'MBPPPlus/[0-9]+', q) or q in found:
            raise ValueError('invalid or duplicate task ID')
        r = {'task_id': q}
        for column in HEADER[1:]:
            value = str(raw[column])
            if not re.fullmatch(r'[0-4]', value):
                raise ValueError('counts must be integers in 0..4')
            r[column] = int(value)
        for n in MODELS:
            c, first, coverage, errors, resource = [r[f'{n}_{k}'] for k in FIELDS]
            if (first not in (0, 1) or coverage != int(c > 0) or first > c
                    or c + errors + resource > DRAWS or (c == DRAWS and first != 1)):
                raise ValueError('inconsistent per-task outcomes')
        found[q] = r
    return [found[q] for q in sorted(found)]


def load_rows(repo):
    m = manifest(repo)
    p = contained(repo, m['fixture'])
    if sha(p) != m['fixture_sha256']:
        raise ValueError('SHA mismatch: public task-count fixture')
    with p.open(encoding='utf8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError('CSV header/order mismatch')
        return validate_rows(list(reader))


def summarize(records):
    models = {n: {k: sum(r[f'{n}_{k}'] for r in records) for k in FIELDS} for n in MODELS}
    comparisons = {}
    for seed in (43, 44):
        a, b = models[f'narrow_{seed}'], models[f'broad_{seed}']
        comparisons[str(seed)] = {
            'correct_count_delta': b['correct'] - a['correct'],
            'candidate_delta': (b['correct'] - a['correct']) / (TASKS * DRAWS),
            'first_delta': (b['first'] - a['first']) / TASKS,
            'coverage_delta': (b['coverage'] - a['coverage']) / TASKS,
            'name_error_delta': b['name_error'] - a['name_error'],
        }
    return {'status': 'RECOMPUTED_COUNTS_NOT_NEW_EVALUATION', 'models': models,
            'comparisons': comparisons, 'tasks': TASKS, 'candidates_per_model': TASKS * DRAWS,
            'new_training_runs': 0, 'new_generations': 0, 'new_container_executions': 0}


def load_snapshot(repo):
    m = manifest(repo)
    result = summarize(load_rows(repo))
    if result['models'] != m['expected_models']:
        raise ValueError('computed counts disagree with pinned expected values')
    for seed, c in result['comparisons'].items():
        for key in ('candidate_delta', 'first_delta', 'coverage_delta'):
            if not math.isclose(c[key], m['comparisons'][seed][key], abs_tol=1e-12):
                raise ValueError('paired direction or estimate changed')
    result['release'] = m['release']
    return result


def bootstrap_snapshot(repo):
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError('Bootstrap requires NumPy; install the documented dev extra') from exc
    m = manifest(repo)
    records = load_rows(repo)
    result = {}
    for seed in (43, 44):
        a, b = f'narrow_{seed}', f'broad_{seed}'
        differences = [[(r[b+'_correct']-r[a+'_correct'])/DRAWS,
                        r[b+'_first']-r[a+'_first'], r[b+'_coverage']-r[a+'_coverage']]
                       for r in records]
        x = np.array(differences)
        indices = np.random.default_rng(20261018).integers(0, TASKS, (20000, TASKS))
        bounds = np.quantile(x[indices].mean(axis=1), [.025, .975], axis=0).tolist()
        if not np.allclose(bounds, m['comparisons'][str(seed)]['ci95'], rtol=0., atol=1e-12):
            raise ValueError('recomputed bootstrap differs from pinned evidence')
        result[str(seed)] = {'ci95': bounds, 'resample_unit': 'task', 'replicates': 20000,
                             'sampling_seed': 20261018, 'numpy_version': np.__version__}
    return result


def verify_documents(repo):
    m = manifest(repo)
    files = list(dict.fromkeys(m['documents'] + m['source_reports']))
    broken = []
    for rel in files:
        path = contained(repo, rel)
        if not path.is_file():
            broken.append(rel)
            continue
        # Verify local file targets, not remote URLs or Markdown fragment IDs.
        text = re.sub(r'```.*?```', '', path.read_text(encoding='utf8'), flags=re.S)
        for target in re.findall(r'\[[^\]\n]*\]\(([^)\s]+)\)', text):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            dest = (path.parent / unquote(parsed.path)).resolve()
            if not dest.is_relative_to(Path(repo).resolve()) or not dest.exists():
                broken.append(rel + ' -> ' + target)
    if broken:
        raise ValueError('broken local documentation links: ' + '; '.join(broken))
    return {'documents_checked': len(files), 'broken_links': [],
            'scope': 'local file targets; remote URLs and fragments not checked'}


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]


def audit_artifacts(repo, artifact_root):
    """Owner-only, read-only original outcome/source verification. No code execution."""
    m = manifest(repo)
    load_snapshot(repo)
    base = Path(artifact_root).resolve()
    r = contained(base, 'exp029_eval4g')
    run = contained(base, 'exp028_v3')
    final = read_json(r/'final.json', m['source_final_sha256'])
    freeze = read_json(run/'evaluation/generation_freeze.json', m['source_generation_freeze_sha256'])
    proofs = read_json(run/'four_trainings.json', m['source_four_trainings_sha256'])
    if (final.get('status') != 'complete' or final['candidate_positions_scored'] != 1920
            or freeze['models'] != m['model_adapters'] or len(proofs) != 4):
        raise ValueError('original experiment identity mismatch')
    refs = _jsonl(r/'reference_results.jsonl')
    if sha(r/'reference_results.jsonl') != final['reference_scores_sha256']:
        raise ValueError('SHA mismatch: reference scoring')
    if len(refs) != 96 or len({x['task_id'] for x in refs}) != 96 or not all(x['passed'] is True for x in refs):
        raise ValueError('reference gate is not complete')
    task_rows = {x['task_id']: dict(task_id=x['task_id']) for x in load_rows(repo)}
    validated_positions = 0
    for name in MODELS:
        weight = freeze['models'][name]
        if not re.fullmatch(r'[0-9a-f]{64}', weight):
            raise ValueError('malformed adapter SHA')
        score_file = r/('scored_'+weight[:16]+'.jsonl')
        if sha(score_file) != m['source_score_files_sha256'][weight]:
            raise ValueError('SHA mismatch: original score records')
        gen_file = run/'evaluation'/('generation_'+weight[:16])/'checkpoint.jsonl'
        if sha(gen_file) != freeze['sources'][weight]['generation_sha256']:
            raise ValueError('SHA mismatch: original generated answers')
        generated = {}
        for g in _jsonl(gen_file):
            key = g['task_id'], g['sample_id']
            if key in generated or key[0] not in task_rows or type(key[1]) is not int or key[1] not in range(4):
                raise ValueError('duplicate or unknown generated sample')
            if hashlib.sha256(g['raw_code'].encode()).hexdigest() != g['code_sha256']:
                raise ValueError('raw generated source SHA mismatch')
            expected_seed = int(hashlib.sha256(f'EXP028|{key[0]}|{key[1]}|42'.encode()).hexdigest()[:8], 16) % (2**31-1)
            if g['seed'] != expected_seed:
                raise ValueError('sampling seed changed')
            generated[key] = g
        records = _jsonl(score_file)
        expected = {(q, i) for q in task_rows for i in range(4)}
        if (len(records) != 384 or {(x['task_id'], x['sample_id']) for x in records} != expected
                or set(generated) != expected):
            raise ValueError('original paired samples incomplete')
        by_task = {q: [] for q in task_rows}
        for v in records:
            if (type(v['passed']) is not bool or type(v['resource_limited']) is not bool
                    or v['passed'] != (v['reason'] == 'none')
                    or v['code_sha256'] != generated[v['task_id'], v['sample_id']]['code_sha256']):
                raise ValueError('source and verdict mismatch')
            by_task[v['task_id']].append(v)
        for q, aa in by_task.items():
            aa.sort(key=lambda x: x['sample_id'])
            bits = [int(x['passed']) for x in aa]
            values = [sum(bits), bits[0], int(any(bits)), sum(x['reason']=='NameError' for x in aa),
                      sum(x['resource_limited'] for x in aa)]
            task_rows[q].update({f'{name}_{k}': str(v) for k, v in zip(FIELDS, values)})
        validated_positions += len(records)
    if validate_rows(list(task_rows.values())) != load_rows(repo):
        raise ValueError('public aggregate fixture differs from original outcomes')
    seen = set()
    for proof in proofs:
        arm, seed = proof['arm'], proof['seed']
        if (arm, seed) not in {(a, s) for a in ('narrow', 'broad') for s in (43, 44)} or (arm, seed) in seen:
            raise ValueError('bad training proof key')
        seen.add((arm, seed))
        d = run/f'{arm}_seed{seed}'
        s = read_json(d/'summary.json')
        if (s['status'] != 'real_GPU_GRPO_completed' or s['global_step'] != 32
                or sha(d/'adapter_model.safetensors') != m['model_adapters'][f'{arm}_{seed}']
                or sha(d/'reward_attempts.jsonl') != s['reward_attempt_log_sha256']
                or len(_jsonl(d/'reward_attempts.jsonl')) != 128
                or s['train_sha256'] != sha(run/'data'/arm/'train.jsonl')):
            raise ValueError('training artifact mismatch')
    return {'status': 'OWNER_ARTIFACTS_VERIFIED_READ_ONLY', 'candidate_positions_verified': validated_positions,
            'reference_controls_verified': len(refs), 'model_files_verified': len(proofs),
            'source_final_sha256': m['source_final_sha256'], 'hidden_test_bodies_read': False,
            'new_training_runs': 0, 'new_generations': 0, 'new_container_executions': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Recompute public recorded counts; no dependencies or GPU')
    demo.add_argument('--json', action='store_true')
    verify = sub.add_parser('verify', help='Validate packaged counts, provenance and local document links')
    verify.add_argument('--bootstrap', action='store_true', help='Also recompute intervals using NumPy')
    audit = sub.add_parser('audit', help='Read and verify owner artifacts, never rerun models or containers')
    audit.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            result = audit_artifacts(ROOT, args.artifact_root)
        else:
            result = load_snapshot(ROOT)
            if args.command == 'verify':
                result['documents'] = verify_documents(ROOT)
                if args.bootstrap:
                    result['recomputed_intervals'] = bootstrap_snapshot(ROOT)
        if args.command == 'demo' and not args.json:
            print(result['status'])
            print('第一阶段：复算既有记录，不是新的训练或评测。')
            print('model\tcorrect/384\tfirst/96\tcoverage/96\tNameError\tresource')
            for name, row in result['models'].items():
                print(name+'\t'+'\t'.join(str(row[k]) for k in FIELDS))
            print('主对照 broad - narrow：', json.dumps(result['comparisons'], ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
