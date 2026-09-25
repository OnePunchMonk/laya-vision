"""Train the vlm_head judge: LoRA on the VLM's language layers + Laya typed head, one epoch.

Data mix: A-OKVQA train (choice, one noul per row, image-swap noul pairs) and GenAI-Bench train prompts
(score, soft targets from the human rating histogram). Eval sets are never touched here.
"""
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import torch

from .backends import VLMHeadJudge
from .core import Example
from .tasks import aokvqa, genai_bench


def train_examples(cfg: Dict) -> List[Example]:
    exs = aokvqa("train", rows=cfg["aokvqa_rows"], noul="one", swap_pairs=cfg["swap_pairs"], seed=cfg["seed"])
    exs += genai_bench("train", prompts=cfg["genai_prompts"])
    random.Random(cfg["seed"]).shuffle(exs)
    return exs


def train(cfg: Dict, vlm=None) -> VLMHeadJudge:
    torch.manual_seed(cfg["seed"])
    judge = VLMHeadJudge(cfg["vlm_id"], lora_r=cfg["lora_r"], vlm=vlm, batch_size=cfg["eval_batch_size"])
    judge.vlm.config.use_cache = False
    judge.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    judge.vlm.enable_input_require_grads()

    exs = train_examples(cfg)
    counts: Dict[str, int] = {}
    for e in exs:
        counts[e.task] = counts.get(e.task, 0) + 1
    print("train examples:", counts, flush=True)

    lora = [p for p in judge.vlm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": cfg["lr_lora"]}, {"params": judge.head.parameters(), "lr": cfg["lr_head"]}], weight_decay=0.0)
    bs = cfg["batch_size"]
    steps = math.ceil(len(exs) / bs)
    if cfg.get("max_steps"):
        steps = min(steps, cfg["max_steps"])
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(cfg["seed"])

    batches = [exs[i * bs : (i + 1) * bs] for i in range(steps)]
    pool = ThreadPoolExecutor(4)
    load = lambda b: [e.image_fn() for e in b]
    nxt = pool.submit(load, batches[0])
    judge.train()
    t0, log = time.time(), []
    for s, b in enumerate(batches):
        imgs = nxt.result()
        if s + 1 < len(batches):
            nxt = pool.submit(load, batches[s + 1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = judge.loss(b, imgs, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora + list(judge.head.parameters()), 1.0)
        opt.step()
        sched.step()
        log.append(float(loss))
        if (s + 1) % 25 == 0 or s + 1 == steps:
            print({"step": s + 1, "of": steps, "loss": round(float(np.mean(log[-25:])), 4), "min": round((time.time() - t0) / 60, 1)}, flush=True)
    pool.shutdown()
    judge.train_minutes = round((time.time() - t0) / 60, 1)
    judge.vlm.config.use_cache = True
    judge.save(cfg["out_dir"])
    return judge
