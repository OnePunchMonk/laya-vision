"""Eval and training tasks as lists of `Example`.

- A-OKVQA: `choice` (4 options) and `noul` ("is this proposed answer correct?") about real photos.
- GenAI-Bench: `score` 1..5 "how well does the image match the prompt", against human Likert ratings
  of images from six text-to-image models. Split by prompt index so no prompt is in both train and test.
"""
import random
from typing import List

import numpy as np

from .core import Example, fit_image

AOKVQA = "HuggingFaceM4/A-OKVQA"
GENAI = "BaiqiL/GenAI-Bench"
GENAI_MODELS = ["DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo"]
MATCH_LEVELS = [
    "does not match the description at all",
    "matches the description poorly, most key elements are wrong or missing",
    "matches the description partially, some key elements are wrong or missing",
    "matches the description mostly, with minor errors",
    "matches the description exactly",
]


def _img_loader(ds, i: int, col: str):
    return lambda: fit_image(ds[i][col])


def _unique(choices: List[str]) -> List[str]:
    seen, out = {}, []
    for c in choices:
        c = c.strip()
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else "%s (%d)" % (c, seen[c]))
    return out


def noul_q(question: str, answer: str):
    return {"t": "noul", "crit": None,
            "ins": 'Question about the image: %s Proposed answer: "%s". Is the proposed answer correct?' % (question, answer)}


def aokvqa(split: str, rows: int = 0, noul: str = "all", swap_pairs: float = 0.0, seed: int = 0) -> List[Example]:
    """noul="all": one noul per option (eval). noul="one": one random option, 50% gold (train).
    swap_pairs: probability of adding the same gold-answer noul with another row's image, labeled false."""
    from datasets import load_dataset

    ds = load_dataset(AOKVQA, split=split)
    if rows:
        ds = ds.select(range(min(rows, len(ds))))
    rng = random.Random(seed)
    out = []
    meta = ds.select_columns(["question_id", "question", "choices", "correct_choice_idx"])  # images stay lazy
    qids = meta["question_id"]

    def onehot(j, k):
        return [1.0 if x == j else 0.0 for x in range(k)]

    for i, r in enumerate(meta):
        choices, gold = _unique(r["choices"]), int(r["correct_choice_idx"])
        qid, fn = r["question_id"], _img_loader(ds, i, "image")
        out.append(Example("%s/choice" % qid, qid, fn, {"t": "choice", "ins": r["question"], "crit": {c: None for c in choices}},
                           onehot(gold, len(choices)), "aokvqa_choice", qid))
        opts = range(len(choices)) if noul == "all" else [gold if rng.random() < 0.5 else rng.choice([j for j in range(len(choices)) if j != gold])]
        for j in opts:
            out.append(Example("%s/noul%d" % (qid, j), qid, fn, noul_q(r["question"], choices[j]), onehot(int(j == gold), 2),
                               "aokvqa_noul", qid, meta={"opt": j}))
        if swap_pairs and rng.random() < swap_pairs:
            o = rng.randrange(len(ds) - 1)
            o += o >= i
            out.append(Example("%s/swap_pos" % qid, qid, fn, noul_q(r["question"], choices[gold]), [0.0, 1.0], "aokvqa_swap", qid))
            out.append(Example("%s/swap_neg" % qid, qids[o], _img_loader(ds, o, "image"),
                               noul_q(r["question"], choices[gold]), [1.0, 0.0], "aokvqa_swap", qid))
    return out


def genai_bench(part: str, prompts: int = 0) -> List[Example]:
    """part="train": prompt Index % 4 != 0 (1,200 prompts). part="test": Index % 4 == 0 (400 prompts)."""
    from datasets import load_dataset

    ds = load_dataset(GENAI, split="train")
    meta = ds.select_columns(["Index", "Prompt", "HumanRatings"])
    out, n = [], 0
    for i, r in enumerate(meta):
        if (r["Index"] % 4 == 0) != (part == "test"):
            continue
        n += 1
        if prompts and n > prompts:
            break
        q = {"t": "score", "ins": 'How well does the image match this description: "%s"?' % r["Prompt"], "crit": MATCH_LEVELS}
        for m in GENAI_MODELS:
            ratings = [x for x in (r["HumanRatings"].get(m) or []) if x is not None and 1 <= x <= 5]
            if not ratings:
                continue
            hist = np.bincount(np.array(ratings) - 1, minlength=5).astype(float)
            out.append(Example("genai/%d/%s" % (r["Index"], m), "genai/%d/%s" % (r["Index"], m), _img_loader(ds, i, m), q,
                               list(hist / hist.sum()), "genai_score", str(r["Index"]), float(np.mean(ratings)), {"model": m}))
    return out
