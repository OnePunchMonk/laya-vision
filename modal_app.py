"""Modal entrypoints.

    modal run modal_app.py --exp smoke                  # a few minutes on one L4
    modal run --detach modal_app.py --exp main          # long run, survives closing the terminal
    modal run --detach modal_app.py --exp frozen_text   # ablations, see EXPERIMENTS

A run killed mid-way (preemption, retry) resumes from runs/<name>/last.pt on the volume.

Feature caches, HF downloads and run outputs live on the `laya-vision` volume.
"""
import json
import os

import modal

app = modal.App("laya-vision")
vol = modal.Volume.from_name("laya-vision", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "transformers>=4.49,<5", "datasets>=2.18", "safetensors>=0.4",
        "laya>=0.3.4", "pillow", "numpy", "sentencepiece",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya_vision")
)
common = dict(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface")])


@app.function(gpu="L4", timeout=3600, **common)
def cache_features(split: str, limit: int = 0) -> str:
    from laya_vision.features import cache_split

    path = "/vol/feats/aokvqa_%s%s.pt" % (split, "_%d" % limit if limit else "")
    if os.path.exists(path):
        return path
    os.makedirs("/vol/feats", exist_ok=True)
    cache_split(split, path, limit=limit)
    vol.commit()
    return path


BASE = {
    "init": "laya",               # laya | modernbert (same encoder, fresh head)
    "train_text": True,           # False: only projector, head and scorer train
    "laya_id": "convaiinnovations/laya",
    "laya_subfolder": None,
    "vision_id": "google/siglip2-base-patch16-224",
    "dataset": "HuggingFaceM4/A-OKVQA",
    "batch_size": 32,
    "lr_text": 2e-5,
    "lr_new": 5e-4,
    "warmup_epochs": 0.5,
    "max_epochs": 100,
    "patience": 10,               # epochs without dev improvement before stopping
    "lr_halve_after": 3,          # halve LR every 3 bad epochs
    "min_delta": 0.0,
    "dev_rows": 1000,             # tail of A-OKVQA train, used for all selection
    "seed": 0,
    "limit": 0,
}
EXPERIMENTS = {
    "smoke": {"limit": 400, "max_epochs": 3, "patience": 2, "lr_halve_after": 1},
    "main": {},
    "frozen_text": {"train_text": False},       # how much must the text encoder adapt?
    "no_laya_init": {"init": "modernbert"},     # does Laya's decision pretraining help?
}


@app.function(gpu="L4", timeout=24 * 3600, retries=modal.Retries(max_retries=2, initial_delay=10.0), **common)
def train(cfg: dict) -> dict:
    from laya_vision.train import run

    res = run(cfg, on_epoch=vol.commit)
    if cfg.get("push_repo"):
        res["hf_url"] = push_run(cfg["run"], cfg["push_repo"])
    return res


CARD = """---
license: apache-2.0
base_model: [convaiinnovations/laya, google/siglip2-base-patch16-224]
tags: [laya, vision, decision-model, private-experiment]
---
# laya-vision / {run}

Private experiment: Laya's typed decision head over SigLIP2 image tokens, trained on A-OKVQA.
Code: github.com/OnePunchMonk/laya-vision (private). Not a benchmark.

```json
{summary}
```
"""


def push_run(run_name: str, repo_id: str) -> str:
    """Upload a finished run folder (weights, config, results, metrics, card) to a private HF repo."""
    from huggingface_hub import HfApi

    run_dir = "/vol/runs/%s" % run_name
    res = json.load(open(os.path.join(run_dir, "results.json")))
    keys = ["laya_text_only_zero_shot", "with_image", "shuffled_image", "stopped", "latency", "train_minutes"]
    with open(os.path.join(run_dir, "README.md"), "w") as f:
        f.write(CARD.format(run=run_name, summary=json.dumps({k: res.get(k) for k in keys}, indent=2)))
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=run_dir, repo_id=repo_id, path_in_repo=run_name, commit_message="Add run %s" % run_name)
    return "https://huggingface.co/%s/tree/main/%s" % (repo_id, run_name)


@app.function(timeout=1800, **common)
def push(run_name: str, repo_id: str) -> str:
    return push_run(run_name, repo_id)


@app.local_entrypoint()
def main(exp: str = "main", seed: int = 0, hf_repo: str = "OnePunchMonk101010/laya-vision", no_push: bool = False):
    cfg = {**BASE, **EXPERIMENTS[exp], "seed": seed}
    run_name = exp if seed == 0 else "%s_s%d" % (exp, seed)
    cfg["run"] = run_name
    cfg["out_dir"] = "/vol/runs/%s" % run_name
    cfg["train_feats"] = cache_features.remote("train", cfg["limit"])
    cfg["val_feats"] = cache_features.remote("validation", cfg["limit"] // 4 if cfg["limit"] else 0)
    cfg["push_repo"] = None if (no_push or exp == "smoke") else hf_repo
    res = train.remote(cfg)
    os.makedirs("runs", exist_ok=True)
    with open("runs/%s.json" % run_name, "w") as f:
        json.dump(res, f, indent=2)
    keys = ["laya_text_only_zero_shot", "with_image", "shuffled_image", "stopped", "latency", "train_minutes"]
    print(json.dumps({k: res.get(k) for k in keys}, indent=2))
    if res.get("hf_url"):
        print("pushed (private):", res["hf_url"])
