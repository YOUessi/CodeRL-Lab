from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

from .evaluation import load_tasks


_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def build_prompt(prompt: str, starter_code: str) -> str:
    return (
        "请完成下面的 Python 编程任务。\n"
        "只返回完整的 Python 代码，不要使用 Markdown 代码块，不要解释。\n\n"
        f"任务：\n{prompt}\n\n"
        f"起始代码：\n{starter_code}\n"
    )


def _contains_entry_point(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in tree.body
    )


def extract_python_code(text: str, entry_point: str) -> str:
    """Normalize model output into executable Python without changing semantics."""
    text = text.strip()
    match = _CODE_FENCE.search(text)
    if match:
        candidate = match.group(1).strip()
    else:
        marker = f"def {entry_point}"
        index = text.find(marker)
        candidate = text[index:].strip() if index >= 0 else text

    if _contains_entry_point(candidate, entry_point):
        return candidate

    lines = candidate.splitlines()
    for end in range(len(lines) - 1, 0, -1):
        shortened = "\n".join(lines[:end]).strip()
        if _contains_entry_point(shortened, entry_point):
            return shortened
    return candidate


def generate_predictions(
    *,
    tasks_path: Path,
    output_path: Path,
    model_name: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "model generation dependencies are missing; install with "
            "pip install -e '.[model]'"
        ) from exc

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tasks = load_tasks(tasks_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks.values():
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]

            for sample_id in range(num_samples):
                generator = torch.Generator(device=model.device)
                generator.manual_seed(seed + sample_id)

                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        generator=generator,
                    )

                generated_tokens = output[0, prompt_length:]
                raw_completion = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )
                completion = extract_python_code(
                    raw_completion,
                    entry_point=task.entry_point,
                )

                handle.write(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "sample_id": sample_id,
                            "completion": completion,
                            "raw_completion": raw_completion,
                            "generation": {
                                "model": model_name,
                                "seed": seed + sample_id,
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_new_tokens": max_new_tokens,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CodeRL-Lab candidates")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_predictions(
        tasks_path=args.tasks,
        output_path=args.output,
        model_name=args.model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
