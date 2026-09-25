"""Pilot training: Laya (English, ModernBERT-large) + frozen SigLIP2 on A-OKVQA.

Loss is Laya's strictly proper reward (log + spherical score), negated. With a deterministic
reported distribution, maximizing expected proper reward reduces to minimizing this directly.
"""
import json
import math
import os
import random
import statistics
import time
from typing import Dict, List

import numpy as np
import torch
from laya.common import QTYPES, ece_score, proper_reward

from .data import build_items
from .model import VisionDecisionModel, load_laya
from .sequence import collate


def batches(items: List[Dict], bs: int, shuffle: bool, seed: int = 0):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i : i + bs]]


def forward(model, b, device, use_image: bool):
    kw = {}
    if use_image:
        kw = {"img_start": b["img_start"].to(device), "img_feats": b["img_feats"].to(device)}
    return model(
        b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
        b["marker_mask"].to(device), b["qtype"].to(device), **kw,
    )


@torch.no_grad()
def evaluate(model, items, feats, pad_id, device, use_image=True, bs=64) -> Dict:
    model.eval()
    rows = []
    for chunk in batches(items, bs, shuffle=False):
        b = collate(chunk, pad_id, feats if use_image else None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(forward(model, b, device, use_image), -1).float().cpu().numpy()
        for it, pi in zip(chunk, p):
            rows.append((it, pi[: len(it["markers"])]))

    ch = [(it, p) for it, p in rows if it["kind"] == "choice"]
    no = [(it, p) for it, p in rows if it["kind"] == "noul"]
    ch_correct = np.array([p.argmax() == it["label"] for it, p in ch], dtype=float)
    ch_conf = np.array([p.max() for _, p in ch])
    no_pred = np.array([p[1] > 0.5 for _, p in no])
    no_gold = np.array([it["label"] == 1 for it, _ in no])
    tpr = (no_pred & no_gold).sum() / max(1, no_gold.sum())
    tnr = (~no_pred & ~no_gold).sum() / max(1, (~no_gold).sum())

    # choice recovered from noul: pick the option with the highest p(true)
    by_q: Dict[str, List] = {}
    for it, p in no:
        by_q.setdefault(it["qid"], []).append((p[1], it["label"]))
    via_noul = np.mean([max(v)[1] == 1 for v in by_q.values()]) if by_q else float("nan")

    return {
        "n_choice": len(ch),
        "choice_acc": float(ch_correct.mean()),
        "choice_ece": ece_score(ch_conf, ch_correct),
        "noul_acc": float((no_pred == no_gold).mean()) if no else float("nan"),
        "noul_balanced_acc": float((tpr + tnr) / 2) if no else float("nan"),
        "choice_via_noul_acc": float(via_noul),
    }


def bootstrap_ci(correct: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def run(cfg: Dict) -> Dict:
    from datasets import load_dataset

    device = "cuda"
    torch.manual_seed(cfg["seed"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    decision, tok, lcfg = load_laya(cfg["laya_id"], cfg.get("laya_subfolder"), device)
    max_len, head_max_len = lcfg.get("max_len", 512), lcfg.get("head_max_len", 192)

    tr_cache, va_cache = torch.load(cfg["train_feats"]), torch.load(cfg["val_feats"])
    tr_feats, va_feats = tr_cache["feats"].float(), va_cache["feats"].float()
    n_img, vdim = tr_feats.shape[1], tr_feats.shape[2]

    cols = ["question_id", "question", "choices", "correct_choice_idx"]
    tr_rows = load_dataset(cfg["dataset"], split="train").select_columns(cols)
    va_rows = load_dataset(cfg["dataset"], split="validation").select_columns(cols)
    if cfg.get("limit"):
        tr_rows = tr_rows.select(range(min(cfg["limit"], len(tr_rows))))
        va_rows = va_rows.select(range(min(cfg["limit"] // 4 or 1, len(va_rows))))
    assert list(tr_rows["question_id"]) == tr_cache["question_id"][: len(tr_rows)], "train feature cache out of order"
    assert list(va_rows["question_id"]) == va_cache["question_id"][: len(va_rows)], "val feature cache out of order"

    tr_items = build_items(tr_rows, tok, n_img, train=True, seed=cfg["seed"], max_len=max_len, head_max_len=head_max_len)
    va_items = build_items(va_rows, tok, n_img, train=False, max_len=max_len, head_max_len=head_max_len)
    va_blind = build_items(va_rows, tok, 0, train=False, max_len=max_len, head_max_len=head_max_len)
    print("items: train %d, val %d" % (len(tr_items), len(va_items)), flush=True)

    model = VisionDecisionModel(decision, vdim, n_img).to(device)
    pad = tok.pad_token_id
    results = {"config": cfg, "n_img_tokens": n_img}

    # Baseline: released Laya, question and options only, no image. What the text prior alone gets.
    results["laya_text_only_zero_shot"] = evaluate(model, va_blind, None, pad, device, use_image=False)
    print("zero-shot text-only:", results["laya_text_only_zero_shot"], flush=True)

    new = [p for n, p in model.named_parameters() if not n.startswith("decision.")]
    old = [p for n, p in model.named_parameters() if n.startswith("decision.")]
    opt = torch.optim.AdamW([{"params": new, "lr": cfg["lr_new"]}, {"params": old, "lr": cfg["lr_text"]}], weight_decay=0.01)
    steps = cfg["epochs"] * math.ceil(len(tr_items) / cfg["batch_size"])
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )

    step, t0, log = 0, time.time(), []
    for ep in range(cfg["epochs"]):
        model.train()
        for chunk in batches(tr_items, cfg["batch_size"], shuffle=True, seed=cfg["seed"] + ep):
            b = collate(chunk, pad, tr_feats)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward(model, b, device, True)
            mask = b["marker_mask"].to(device).float()
            p = torch.softmax(logits.float(), -1)
            loss = -proper_reward(p, b["target"].to(device), b["qtype"].to(device), mask).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                log.append({"step": step, "loss": float(loss), "elapsed_s": round(time.time() - t0, 1)})
                print(log[-1], flush=True)
        ev = evaluate(model, va_items, va_feats, pad, device)
        print("epoch %d val:" % ep, ev, flush=True)
        results.setdefault("per_epoch", []).append(ev)

    results["train_log"] = log
    results["train_minutes"] = round((time.time() - t0) / 60, 2)
    results["with_image"] = evaluate(model, va_items, va_feats, pad, device)
    # Ablation: each question gets another question's image. A drop here means the image is used.
    perm = torch.randperm(len(va_feats), generator=torch.Generator().manual_seed(1))
    results["shuffled_image"] = evaluate(model, va_items, va_feats[perm], pad, device)

    ch = [it for it in va_items if it["kind"] == "choice"]
    b_all = []
    model.eval()
    with torch.no_grad():
        for chunk in batches(ch, 64, False):
            b = collate(chunk, pad, va_feats)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = forward(model, b, device, True).argmax(-1).cpu()
            b_all.extend((pred == b["label"]).tolist())
    results["with_image"]["choice_acc_ci95"] = bootstrap_ci(np.array(b_all, dtype=float))

    results["latency"] = bench_latency(model, tok, cfg, max_len, head_max_len, device)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    from safetensors.torch import save_file

    save_file({k: v.half().contiguous() for k, v in model.state_dict().items()}, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "laya_vision_config.json"), "w") as f:
        json.dump({"laya": lcfg, "laya_id": cfg["laya_id"], "vision_id": cfg["vision_id"], "n_img": n_img, "vision_dim": vdim, "pool": 2}, f, indent=2)
    return results


@torch.no_grad()
def bench_latency(model, tok, cfg, max_len, head_max_len, device, runs: int = 30) -> Dict:
    """End to end on raw pixels: SigLIP2 forward + one decision forward, 1 and 10 questions per image."""
    from datasets import load_dataset

    from .data import noul_question
    from .features import encode_images, load_vision
    from .sequence import build_vision_sequence

    proc, vm = load_vision(cfg["vision_id"], device)
    img = load_dataset(cfg["dataset"], split="validation").select(range(1))[0]["image"]
    model.eval()
    out = {"gpu": torch.cuda.get_device_name()}
    for nq in (1, 10):
        items = []
        for k in range(nq):
            ids, markers, start = build_vision_sequence(tok, noul_question("What is shown?", "option %d" % k), model.n_img, max_len=max_len, head_max_len=head_max_len)
            items.append({"ids": ids, "markers": markers, "img_start": start, "img_row": 0, "qtype": QTYPES["noul"]})
        times = []
        for r in range(runs + 5):
            torch.cuda.synchronize()
            t = time.perf_counter()
            f = encode_images([img], proc, vm).float()
            b = collate(items, tok.pad_token_id, f)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forward(model, b, device, True)
            torch.cuda.synchronize()
            if r >= 5:
                times.append((time.perf_counter() - t) * 1000)
        out["p50_ms_%dq" % nq] = round(statistics.median(times), 1)
    return out
