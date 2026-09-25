"""One evaluator for every judge.

Per task: accuracy / ECE (choice, noul), Spearman and within-prompt pairwise accuracy against human
ratings (score), and a shuffled-image check: the same questions with other examples' images. A judge
that barely changes under shuffling is answering from the text, not the image.
"""
import random
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
from laya.common import ece_score

from .core import Example


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _pairwise(groups: Dict[str, List], min_gap: float = 0.0) -> float:
    """Within each prompt, fraction of image pairs the judge orders like the human mean. Ties count half."""
    hit = tot = 0.0
    for items in groups.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (pi, hi), (pj, hj) = items[i], items[j]
                if abs(hi - hj) <= min_gap:
                    continue
                tot += 1
                hit += 0.5 if pi == pj else float((pi > pj) == (hi > hj))
    return hit / tot if tot else float("nan")


def task_metrics(examples: List[Example], probs: List[np.ndarray]) -> Dict:
    t = examples[0].q["t"]
    gold = [int(np.argmax(e.gold)) for e in examples]
    pred = [int(np.argmax(p)) for p in probs]
    correct = np.array([g == p for g, p in zip(gold, pred)], dtype=float)
    out = {"n": len(examples)}
    if t in ("choice", "noul"):
        out["acc"] = float(correct.mean())
        out["ece"] = ece_score(np.array([p.max() for p in probs]), correct)
    if t == "noul":
        g = np.array(gold) == 1
        pr = np.array(pred) == 1
        out["balanced_acc"] = float(((pr & g).sum() / max(1, g.sum()) + (~pr & ~g).sum() / max(1, (~g).sum())) / 2)
        by = defaultdict(list)
        for e, p in zip(examples, probs):
            if "opt" in e.meta:
                by[e.group].append((p[1], e.gold[1]))
        if by:
            out["choice_via_noul_acc"] = float(np.mean([max(v)[1] == 1 for v in by.values()]))
    if t == "score":
        exp = np.array([float((np.arange(1, len(p) + 1) * p).sum()) for p in probs])
        hum = np.array([e.human for e in examples])
        out["spearman"] = _spearman(exp, hum)
        out["pearson"] = float(np.corrcoef(exp, hum)[0, 1])
        groups = defaultdict(list)
        for e, s in zip(examples, exp):
            groups[e.group].append((s, e.human))
        out["pairwise_acc"] = _pairwise(groups)
        out["pairwise_acc_gap1"] = _pairwise(groups, min_gap=1.0)
        out["log_score_vs_humans"] = float(np.mean([(np.array(e.gold) * np.log(np.clip(p, 1e-9, 1))).sum() for e, p in zip(examples, probs)]))
        out["mean_expected_score"] = float(exp.mean())
    return out


def evaluate(judge, tasks: Dict[str, List[Example]], shuffle_tasks=("aokvqa_choice", "genai_score"), seed: int = 0) -> Dict:
    res = {"judge": judge.name}
    for name, exs in tasks.items():
        t0 = time.time()
        probs = judge.predict(exs)
        m = task_metrics(exs, probs)
        m["examples_per_s"] = round(len(exs) / (time.time() - t0), 2)
        if name in shuffle_tasks:
            # derangement over distinct images so no example keeps its own image
            keys = sorted({e.image_key for e in exs})
            perm = keys[:]
            random.Random(seed).shuffle(perm)
            perm = perm[1:] + perm[:1]
            to = dict(zip(keys, perm))
            fn_of = {e.image_key: e.image_fn for e in exs}
            sp = judge.predict(exs, image_fns=[fn_of[to[e.image_key]] for e in exs], image_keys=[to[e.image_key] for e in exs])
            sm = task_metrics(exs, sp)
            m["shuffled"] = {k: v for k, v in sm.items() if k in ("acc", "spearman", "pairwise_acc", "mean_expected_score")}
            if "acc" in m:
                m["image_gain_acc"] = m["acc"] - sm["acc"]
            if "mean_expected_score" in m:
                m["score_drop_when_shuffled"] = m["mean_expected_score"] - sm["mean_expected_score"]
        print(name, m, flush=True)
        res[name] = m
    return res


def latency(judge, examples: List[Example], runs: int = 20) -> Dict:
    """Single-image latency, all questions for that image in one call, on already-decoded pixels."""
    import torch

    img = examples[0].image_fn()
    exs = [e for e in examples if e.image_key == examples[0].image_key]
    times = []
    for r in range(runs + 3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        judge.predict(exs, image_fns=[lambda: img] * len(exs), image_keys=["__lat__%d" % r] * len(exs))
        torch.cuda.synchronize()
        if r >= 3:
            times.append((time.perf_counter() - t) * 1000)
    return {"questions": len(exs), "p50_ms": round(float(np.median(times)), 1)}
