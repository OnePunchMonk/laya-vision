"""Training: Laya (English, ModernBERT-large) + frozen SigLIP2 on A-OKVQA.

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


def load_text_side(cfg: Dict, device: str):
    """`laya` starts from the released checkpoint; `modernbert` is the same encoder with a fresh head (ablation)."""
    if cfg.get("init", "laya") == "laya":
        return load_laya(cfg["laya_id"], cfg.get("laya_subfolder"), device)
    from laya.common import build_model
    from transformers import AutoTokenizer

    lcfg = {"encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "act_costs": {"escalate": 0.5},
            "max_len": 512, "head_max_len": 192}
    model = build_model(lcfg).to(device)
    model.encoder.config.reference_compile = False
    return model, AutoTokenizer.from_pretrained(lcfg["encoder"]), lcfg


def run(cfg: Dict, on_epoch=None) -> Dict:
    """Train with per-epoch dev eval, LR halving on plateau, early stopping and resumable checkpoints.

    The dev split is the last `dev_rows` rows of A-OKVQA train and drives every selection decision.
    The validation split is evaluated once, on the best checkpoint, after training ends.
    """
    from datasets import load_dataset
    from safetensors.torch import load_file, save_file

    device = "cuda"
    torch.manual_seed(cfg["seed"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    last_path, best_path = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "model.safetensors")
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    decision, tok, lcfg = load_text_side(cfg, device)
    max_len, head_max_len = lcfg.get("max_len", 512), lcfg.get("head_max_len", 192)

    tr_cache, va_cache = torch.load(cfg["train_feats"]), torch.load(cfg["val_feats"])
    tr_feats, va_feats = tr_cache["feats"].float(), va_cache["feats"].float()
    n_img, vdim = tr_feats.shape[1], tr_feats.shape[2]

    cols = ["question_id", "question", "choices", "correct_choice_idx"]
    all_tr = load_dataset(cfg["dataset"], split="train").select_columns(cols)
    va_rows = load_dataset(cfg["dataset"], split="validation").select_columns(cols)
    if cfg.get("limit"):
        all_tr = all_tr.select(range(min(cfg["limit"], len(all_tr))))
        va_rows = va_rows.select(range(min(cfg["limit"] // 4 or 1, len(va_rows))))
    assert list(all_tr["question_id"]) == tr_cache["question_id"][: len(all_tr)], "train feature cache out of order"
    assert list(va_rows["question_id"]) == va_cache["question_id"][: len(va_rows)], "val feature cache out of order"

    n_dev = min(cfg["dev_rows"], len(all_tr) // 5)
    n_tr = len(all_tr) - n_dev
    tr_rows = [all_tr[i] for i in range(n_tr)]
    dev_rows = [all_tr[i] for i in range(n_tr, len(all_tr))]
    dev_feats = tr_feats[n_tr:]
    kw = dict(max_len=max_len, head_max_len=head_max_len)
    dev_items = build_items(dev_rows, tok, n_img, train=False, **kw)
    va_items = build_items(va_rows, tok, n_img, train=False, **kw)
    va_blind = build_items(va_rows, tok, 0, train=False, **kw)
    print("rows: train %d, dev %d, val %d" % (n_tr, n_dev, len(va_rows)), flush=True)

    model = VisionDecisionModel(decision, vdim, n_img).to(device)
    if not cfg.get("train_text", True):
        for p in model.decision.encoder.parameters():
            p.requires_grad_(False)
    pad = tok.pad_token_id

    new = [p for n, p in model.named_parameters() if not n.startswith("decision.") and p.requires_grad]
    old = [p for n, p in model.named_parameters() if n.startswith("decision.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": new, "lr": cfg["lr_new"]}, {"params": old, "lr": cfg["lr_text"]}], weight_decay=0.01)
    steps_per_epoch = math.ceil(2 * n_tr / cfg["batch_size"])
    warm = max(1, int(cfg.get("warmup_epochs", 0.5) * steps_per_epoch))
    base_lrs = [g["lr"] for g in opt.param_groups]

    state = {"epoch": 0, "step": 0, "best": -1.0, "best_epoch": -1, "bad": 0, "lr_scale": 1.0, "t_train": 0.0}
    results = {"config": cfg, "n_img_tokens": n_img, "rows": {"train": n_tr, "dev": n_dev, "val": len(va_rows)}}
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state, results = ck["state"], ck["results"]
        print("resumed at epoch %d (best dev %.4f @ %d)" % (state["epoch"], state["best"], state["best_epoch"]), flush=True)
    else:
        # Baseline: released Laya, question and options only. The text-prior floor.
        results["laya_text_only_zero_shot"] = evaluate(model, va_blind, None, pad, device, use_image=False)
        print("zero-shot text-only:", results["laya_text_only_zero_shot"], flush=True)

    while state["epoch"] < cfg["max_epochs"] and state["bad"] < cfg["patience"]:
        ep, t0 = state["epoch"], time.time()
        # fresh option order and noul candidates every epoch
        tr_items = build_items(tr_rows, tok, n_img, train=True, seed=cfg["seed"] * 1000 + ep, **kw)
        model.train()
        losses = []
        for chunk in batches(tr_items, cfg["batch_size"], shuffle=True, seed=cfg["seed"] * 1000 + ep):
            s = state["step"]
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * state["lr_scale"] * min(1.0, (s + 1) / warm)
            b = collate(chunk, pad, tr_feats)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward(model, b, device, True)
            p = torch.softmax(logits.float(), -1)
            loss = -proper_reward(p, b["target"].to(device), b["qtype"].to(device), b["marker_mask"].to(device).float()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            state["step"] += 1
            losses.append(float(loss))

        dev = evaluate(model, dev_items, dev_feats, pad, device)
        score = dev["choice_acc"]
        improved = score > state["best"] + cfg.get("min_delta", 0.0)
        if improved:
            state.update(best=score, best_epoch=ep, bad=0)
            save_file({k: v.half().contiguous() for k, v in model.state_dict().items()}, best_path)
        else:
            state["bad"] += 1
            if state["bad"] % cfg["lr_halve_after"] == 0:
                state["lr_scale"] *= 0.5
        state["epoch"] += 1
        state["t_train"] += time.time() - t0
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "dev": dev, "best_dev": state["best"],
               "best_epoch": state["best_epoch"], "bad": state["bad"], "lr_scale": state["lr_scale"],
               "epoch_min": round((time.time() - t0) / 60, 2)}
        print("epoch", json.dumps(rec), flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "results": results}, last_path)
        if on_epoch:
            on_epoch()

    results["stopped"] = {"epochs_run": state["epoch"], "best_epoch": state["best_epoch"], "best_dev_choice_acc": state["best"],
                          "reason": "patience" if state["bad"] >= cfg["patience"] else "max_epochs"}
    results["train_minutes"] = round(state["t_train"] / 60, 1)

    # One untouched evaluation on validation, with the dev-selected checkpoint.
    model.load_state_dict({k: v.float() for k, v in load_file(best_path).items()})
    results["with_image"] = evaluate(model, va_items, va_feats, pad, device)
    perm = torch.randperm(len(va_feats), generator=torch.Generator().manual_seed(1))
    results["shuffled_image"] = evaluate(model, va_items, va_feats[perm], pad, device)
    correct = []
    model.eval()
    with torch.no_grad():
        for chunk in batches([it for it in va_items if it["kind"] == "choice"], 64, False):
            b = collate(chunk, pad, va_feats)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                correct.extend((forward(model, b, device, True).argmax(-1).cpu() == b["label"]).tolist())
    results["with_image"]["choice_acc_ci95"] = bootstrap_ci(np.array(correct, dtype=float))
    results["latency"] = bench_latency(model, tok, cfg, max_len, head_max_len, device)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "laya_vision_config.json"), "w") as f:
        json.dump({"laya": lcfg, "init": cfg.get("init", "laya"), "laya_id": cfg["laya_id"], "vision_id": cfg["vision_id"],
                   "n_img": n_img, "vision_dim": vdim, "pool": 2}, f, indent=2)
    if os.path.exists(last_path):
        os.remove(last_path)  # optimizer state is ~5 GB; the best weights are what matter now
    if on_epoch:
        on_epoch()
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
