from coderl_lab.analysis.exp7b_fence_recheck import code_without_outer_fence, can_parse_python

def test_complete_python_fence_only():
    tick=chr(96)*3
    raw=tick+"python\nx=1\n"+tick+"\n\nThis is an explanation."
    code,kind=code_without_outer_fence(raw)
    assert (code,kind)==("x=1","first_closed_python_fence")
    assert can_parse_python(code)

def test_missing_closing_fence_is_not_silently_repaired():
    tick=chr(96)*3
    raw=tick+"python\nx=1\n"
    code,kind=code_without_outer_fence(raw)
    assert code==raw and kind=="unclosed_fence_unchanged"
    assert not can_parse_python(code)

def test_plain_code_unchanged():
    raw="def add(a,b):\n    return a+b\n"
    code,kind=code_without_outer_fence(raw)
    assert code==raw and kind=="no_leading_python_fence"

def test_invalid_code_inside_fence_is_not_semantically_corrected():
    tick=chr(96)*3
    raw=tick+"python\nif:\n    pass\n"+tick
    code,kind=code_without_outer_fence(raw)
    assert kind=="first_closed_python_fence"
    assert code=="if:\n    pass"
    assert not can_parse_python(code)
