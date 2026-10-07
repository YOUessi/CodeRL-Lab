from coderl_lab.data.runtime_repair_preferences import (
    apply_import_repair,
    import_lines_for,
)


def test_repairability_import_mapping() -> None:
    assert import_lines_for(["re"]) == ["import re"]
    assert import_lines_for(["Counter"]) == [
        "from collections import Counter"
    ]
    assert import_lines_for(["not_a_stdlib_symbol"]) is None


def test_apply_import_repair_is_prefix_only() -> None:
    code = "def f(x):\n    return re.search('a', x)"
    patched = apply_import_repair(code, ["import re"])
    assert patched == "import re\n" + code
