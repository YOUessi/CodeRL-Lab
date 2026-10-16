"""EXP-022 audit invariants: tiny synthetic paired records, no code execution."""
import json
from pathlib import Path
import pytest
from coderl_lab.analysis.exp022_flip_audit import indexed,parse_features,SEEDS,DRAWS,EXPECTED_TASKS

def test_duplicate_key_fails():
    row={"task_id":"a","sample_id":0,"arm":"sft"}
    with pytest.raises(ValueError,match="duplicate"):
        indexed([row,row])

def test_ast_classification():
    same=parse_features("def f(x):\n return x + 1")
    different=parse_features("def f(x):\n if x: return x\n return 0")
    assert same["ast_valid"] and different["ast_valid"]
    assert same["ast_dump_hash"]!=different["ast_dump_hash"]
    assert same["control_flow"]==0 and different["control_flow"]==1
    assert not parse_features("def broken(")["ast_valid"]

def test_expected_audit_dimensions():
    assert SEEDS==(42,43,44)
    assert DRAWS==4
    assert EXPECTED_TASKS==96
