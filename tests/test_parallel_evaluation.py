from coderl_lab.evaluation import evaluate_predictions
from coderl_lab.execution import PythonExecutor
from coderl_lab.schema import CodeTask, TestCase as CodeTestCase


def make_tasks():
    task = CodeTask(
        task_id="add",
        prompt="add",
        entry_point="add",
        starter_code="def add(a,b):\n    pass\n",
        public_tests=(CodeTestCase(args=[1, 2], expected=3),),
        hidden_tests=(CodeTestCase(args=[-1, 1], expected=0),),
    )
    return {"add": task}


def make_predictions():
    return [
        {
            "task_id": "add",
            "sample_id": 0,
            "completion": "def add(a,b):\n    return a+b",
        },
        {
            "task_id": "add",
            "sample_id": 1,
            "completion": "def add(a,b):\n    return a-b",
        },
    ]


def test_parallel_matches_serial() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    serial = evaluate_predictions(
        tasks=make_tasks(),
        predictions=make_predictions(),
        executor=executor,
        ks=(1, 2),
        max_workers=1,
    )
    parallel = evaluate_predictions(
        tasks=make_tasks(),
        predictions=make_predictions(),
        executor=executor,
        ks=(1, 2),
        max_workers=2,
    )
    assert serial == parallel
