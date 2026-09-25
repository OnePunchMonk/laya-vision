import numpy as np
import pytest

from laya_vision.judge.core import Example
from laya_vision.judge.evaluate import _pairwise, task_metrics
from laya_vision.judge.tasks import MATCH_LEVELS


def ex(q, gold, **kw):
    return Example(kw.pop("key", "k"), kw.pop("image_key", "i"), lambda: None, q, gold, **kw)


def test_pairwise_counts_ties_half():
    groups = {"p": [(3.0, 4.0), (2.0, 1.0), (2.0, 3.0)]}
    # pairs: (3,2)vs(4,1) right; (3,2)vs(4,3) right; (2,2) tie vs (1,3) -> 0.5
    assert _pairwise(groups) == pytest.approx(2.5 / 3)


def test_score_metrics_follow_expected_score():
    q = {"t": "score", "ins": "match?", "crit": MATCH_LEVELS}
    exs = [ex(q, [0, 0, 0, 0, 1.0], group="a", human=5.0), ex(q, [1.0, 0, 0, 0, 0], group="a", human=1.0),
           ex(q, [0, 0, 1.0, 0, 0], group="b", human=3.0)]
    good = [np.eye(5)[4], np.eye(5)[0], np.eye(5)[2]]
    m = task_metrics(exs, good)
    assert m["spearman"] == pytest.approx(1.0) and m["pairwise_acc"] == 1.0
    swapped = [good[1], good[0], good[2]]
    assert task_metrics(exs, swapped)["pairwise_acc"] == 0.0


def test_noul_choice_via_noul():
    q = {"t": "noul", "ins": "correct?", "crit": None}
    exs = [ex(q, [0, 1.0], group="g", meta={"opt": 0}), ex(q, [1.0, 0], group="g", meta={"opt": 1})]
    m = task_metrics(exs, [np.array([0.3, 0.7]), np.array([0.6, 0.4])])
    assert m["acc"] == 1.0 and m["choice_via_noul_acc"] == 1.0 and m["balanced_acc"] == 1.0


def test_vlm_head_prompt_has_one_marker_per_option():
    from transformers import AutoTokenizer

    from laya_vision.judge.backends import MARK, LETTERS, VLMHeadJudge

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    mark = tok.convert_tokens_to_ids(MARK)
    for q in [{"t": "choice", "ins": "Color?", "crit": {"red": None, "blue": None, "a red-blue mix": None}},
              {"t": "noul", "ins": "Is it red?", "crit": None},
              {"t": "score", "ins": "match?", "crit": MATCH_LEVELS}]:
        text = VLMHeadJudge.prompt(None, q, order=None)
        n = {"choice": 3, "noul": 2, "score": 5}[q["t"]]
        assert tok(text)["input_ids"].count(mark) == n
    assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in list(LETTERS[:8]) + ["Yes", "No", "1", "5"])
