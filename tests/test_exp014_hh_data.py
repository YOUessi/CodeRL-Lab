import pytest

from coderl_lab.datasets.exp014_hh_helpful import (
    parse_pair, choose, NUM_PAIRS, normalized_text,
)


def entry(prompt="What is the time complexity of binary search?", good="O(log n).", bad="O(n²)."):
    prefix="\n\nHuman: "+prompt+"\n\nAssistant:"
    return {"chosen":prefix+" "+good, "rejected":prefix+" "+bad}


class MockTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return " ".join(m["content"] for m in messages)
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":list(range(len(text.split())))}


def test_parsing_shared_context_never_selects_label_from_prompts():
    p=parse_pair(entry())
    assert p is not None
    assert len(p["messages"])==1
    assert p["messages"][0]["role"]=="user"
    assert "O(log n)" not in p["messages"][0]["content"]
    assert p["chosen"]=="O(log n)."
    assert p["rejected"]=="O(n²)."


def test_reject_misaligned_or_identical_answer_pairs():
    bad=entry()
    bad["rejected"]=bad["rejected"].replace("binary search","linear search")
    assert parse_pair(bad) is None
    assert parse_pair(entry(good="same",bad="same")) is None
    assert parse_pair({"chosen":"text"}) is None


def test_multiturn_and_alternating_roles():
    pre="\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: Can you summarize?"
    parsed=parse_pair({"chosen":pre+"\n\nAssistant: Yes.",
                       "rejected":pre+"\n\nAssistant: No."})
    assert [m["role"] for m in parsed["messages"]]==["user","assistant","user"]
    wrong="\n\nHuman: hi\n\nHuman: again"
    assert parse_pair({"chosen":wrong+"\n\nAssistant: a",
                       "rejected":wrong+"\n\nAssistant: b"}) is None


def test_seeded_cohort_is_stable_not_answer_dependent():
    source=[entry(prompt=f"Unique question number {i} will be solved") for i in range(NUM_PAIRS+5)]
    x,stats=choose(source,MockTokenizer(),set())
    reversed_source=list(reversed(source))
    y,_=choose(reversed_source,MockTokenizer(),set())
    assert len(x)==NUM_PAIRS
    assert [r["task_id"] for r in x]==[r["task_id"] for r in y]
    source=[entry(prompt="Known previous preference prompt")]+source
    _,stat=choose(source,MockTokenizer(),{normalized_text("Known previous preference prompt")})
    assert stat["overlap_with_previous_train_or_val"]==1
