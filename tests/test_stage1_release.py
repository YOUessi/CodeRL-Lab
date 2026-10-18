"""Stage-one delivery contracts: no candidate execution, model loading or network."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('sft','narrow_43','broad_43','narrow_44','broad_44')
FIELDS = ('correct','first','coverage','name_error','resource')


def runner():
    p = ROOT/'scripts/stage1_release.py'
    assert p.is_file(), 'release runner not implemented'
    spec = importlib.util.spec_from_file_location('stage1_release_tested', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def records():
    return [dict(task_id=f'MBPPPlus/{i}', **{f'{n}_{k}': '0' for n in MODELS for k in FIELDS}) for i in range(96)]


def test_actual_snapshot_matches_verified_counts():
    m = runner(); result = m.load_snapshot(ROOT)
    assert result['models']['sft']['correct'] == 202
    assert result['models']['broad_44']['correct'] == 207
    assert result['comparisons']['43']['correct_count_delta'] == -1
    assert result['comparisons']['44']['correct_count_delta'] == 4
    assert result['new_training_runs'] == result['new_generations'] == 0


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unknown_column', 'non_integer', 'bad_first', 'contradiction', 'excess_errors'])
def test_corrupt_counts_rejected(kind):
    m = runner(); rr = records()
    if kind == 'missing': rr.pop()
    if kind == 'duplicate': rr[-1]['task_id'] = rr[0]['task_id']
    if kind == 'unknown_column': rr[0]['raw_code'] = 'forbidden'
    if kind == 'non_integer': rr[0]['sft_correct'] = 'nan'
    if kind == 'bad_first': rr[0]['sft_first'] = '2'
    if kind == 'contradiction': rr[0]['sft_first'] = '1'
    if kind == 'excess_errors': rr[0].update(sft_name_error='3', sft_resource='2')
    with pytest.raises(ValueError): m.validate_rows(rr)


def test_valid_zero_fixture_and_metric_orientation():
    m = runner(); rr = records(); rr[0].update(broad_44_correct='1', broad_44_coverage='1')
    result = m.summarize(m.validate_rows(rr))
    assert result['comparisons']['44']['correct_count_delta'] == 1
    assert result['comparisons']['43']['correct_count_delta'] == 0


def test_csv_tampering_rejected(tmp_path):
    m = runner()
    shutil.copytree(ROOT/'releases', tmp_path/'releases')
    shutil.copytree(ROOT/'results/stage1', tmp_path/'results/stage1')
    path = tmp_path/'results/stage1/coverage_task_counts.csv'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='SHA'): m.load_snapshot(tmp_path)


def test_main_demo_no_site_packages():
    m = runner()
    p = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/stage1_release.py'), 'demo'], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert 'broad_44' in p.stdout and '207' in p.stdout
    assert 'RECOMPUTED_COUNTS_NOT_NEW_EVALUATION' in p.stdout


def test_missing_owner_artifacts_fail_without_writes(tmp_path):
    m = runner()
    with pytest.raises((FileNotFoundError, ValueError)): m.audit_artifacts(ROOT, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_release_documents_have_no_broken_local_links():
    m = runner()
    assert m.verify_documents(ROOT)['broken_links'] == []


def test_bootstrap_reproduces_stored_task_intervals():
    m = runner(); report = m.bootstrap_snapshot(ROOT)
    assert report['43']['ci95'][0][0] == pytest.approx(-0.013020833333333334)
    assert report['43']['ci95'][1][0] == pytest.approx(0.0078125)
    assert report['44']['ci95'][0][0] == pytest.approx(-0.0026041666666666665)
    assert report['44']['ci95'][1][0] == pytest.approx(0.026041666666666668)
