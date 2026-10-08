from pathlib import Path
import json
from coderl_lab.datasets.livecodebench import prepare_public_view


def test_v5_public_only_and_backward_compatible_v6(tmp_path: Path) -> None:
    source = tmp_path / "test5.jsonl"
    row = {
      "question_id":"task-a", "question_title":"Example", "question_content":"Output hello.",
      "platform":"atcoder", "contest_id":"a", "contest_date":"2024-09-01",
      "starter_code":"", "difficulty":"easy",
      "public_test_cases":json.dumps([{"input":"","output":"hello","testtype":"stdin"}]),
      "private_test_cases":"PRIVATE_SENSITIVE_SENTINEL",
      "metadata":json.dumps({}),
    }
    source.write_text(json.dumps(row)+"\n")
    for version in ("v5","v6"):
        directory = tmp_path / version
        kwargs = {"fine_grained_version":version} if version == "v5" else {}
        m = prepare_public_view(source_path=source,output_dir=directory,source_revision="fixture",**kwargs)
        assert m["fine_grained_version"] == version
        assert m["tasks"] == 1
        s = (directory/"public_tasks.jsonl").read_text()
        assert "PRIVATE_SENSITIVE_SENTINEL" not in s
        assert "private_test_cases" not in s
        assert m["private_tests_decoded"] is False
