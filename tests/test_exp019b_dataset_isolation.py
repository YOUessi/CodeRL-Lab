import ast
from coderl_lab.datasets.exp019b_mbppplus_freeze import eligible


def fake(taskid,code="def shared(a, b):\n    return a + b",
         prompt="Write a function combining a and b."):
    return {"task_id":taskid, "code":code, "prompt":prompt,
            "test":"assert False", "test_list":["assert False"]}


def test_exact_source_signature_only_and_no_model_visible_labels():
    entry,reason=eligible(fake(10))
    assert reason=="eligible"
    public=entry["public"]
    assert public["task_id"]=="MBPPPlus/10"
    assert public["entry_point"]=="shared"
    assert public["signature"]=="def shared(a, b):"
    assert set(public)=={"task_id","source_task_id","entry_point","signature","description","messages"}
    visible=str(public)
    assert "return a + b" not in visible
    assert "assert False" not in visible
    assert public["messages"][0]["role"]=="user"


def test_only_static_structure_used_in_selection():
    a,_=eligible(fake(8,code="def shared(a,b):\n    return a+b"))
    b,_=eligible(fake(8,code="def shared(a,b):\n    return a*b"))
    assert a==b


def test_reject_two_function_or_malformed_reference():
    assert eligible(fake(1,code="def f(x):return x\ndef g(x):return x"))[0] is None
    assert eligible(fake(1,code="bad python!"))[0] is None
    assert eligible(fake(1,prompt=""))[0] is None
