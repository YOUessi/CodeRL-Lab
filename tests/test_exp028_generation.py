import hashlib,json
import pytest
from coderl_lab.analysis.exp028_protocol import public_rows,validate_generations,sample_seed


def test_private_columns_rejected(tmp_path):
    p=tmp_path/'p.jsonl';p.write_text(json.dumps({'task_id':'a','messages':[{'role':'user','content':'task'}],'test_cases':[]})+'\n')
    with pytest.raises(ValueError):public_rows(p)


def test_duplicate_public_tasks_rejected(tmp_path):
    p=tmp_path/'p.jsonl';r={'task_id':'a','messages':[{'role':'user','content':'task'}]}
    p.write_text((json.dumps(r)+'\n')*2)
    with pytest.raises(ValueError):public_rows(p)


def test_generation_identity_checks():
    r={'task_id':'a','sample_id':0,'seed':sample_seed('a',0),'raw_code':'pass','code_sha256':hashlib.sha256(b'pass').hexdigest()}
    found,expected=validate_generations([r],[{'task_id':'a'}])
    assert len(found)==1 and len(expected)==4
    with pytest.raises(ValueError):validate_generations([r,r],[{'task_id':'a'}])
    with pytest.raises(ValueError):validate_generations([{**r,'raw_code':'fail'}],[{'task_id':'a'}])
    with pytest.raises(ValueError):validate_generations([{**r,'seed':123}],[{'task_id':'a'}])
