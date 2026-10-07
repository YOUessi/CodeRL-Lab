from coderl_lab.data.runtime_repair_preferences import (
    apply_import_repair,
    import_lines_for,
)


def test_module_import_mapping() -> None:
    assert import_lines_for(["re", "math"]) == [
        "import re",
        "import math",
    ]


def test_symbol_import_mapping() -> None:
    assert import_lines_for(["Counter", "groupby", "ceil"]) == [
        "from collections import Counter",
        "from itertools import groupby",
        "from math import ceil",
    ]


def test_unknown_name_is_rejected() -> None:
    assert import_lines_for(["some_project_helper"]) is None


def test_apply_import_repair_only_prepends_imports() -> None:
    code = "def f(x):\n    return math.sqrt(x)\n"
    patched = apply_import_repair(code, ["import math"])
    assert patched == "import math\ndef f(x):\n    return math.sqrt(x)\n"
