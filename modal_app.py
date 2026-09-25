"""Modal entrypoints.

    modal run modal_app.py --limit 400 --epochs 1   # smoke test, a few minutes on one L4
    modal run modal_app.py                          # pilot: full A-OKVQA train, 3 epochs

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


@app.function(gpu="L4", timeout=4 * 3600, **common)
def train(cfg: dict) -> dict:
    from laya_vision.train import run

    res = run(cfg)
    vol.commit()
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


@app.function(timeout=1800, **common)
def push(run_name: str, repo_id: str) -> str:
    from huggingface_hub import HfApi

    run_dir = "/vol/runs/%s" % run_name
    res = json.load(open(os.path.join(run_dir, "results.json")))
    keys = ["laya_text_only_zero_shot", "with_image", "shuffled_image", "latency", "train_minutes"]
    with open(os.path.join(run_dir, "README.md"), "w") as f:
        f.write(CARD.format(run=run_name, summary=json.dumps({k: res.get(k) for k in keys}, indent=2)))
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=run_dir, repo_id=repo_id, path_in_repo=run_name, commit_message="Add run %s" % run_name)
    return "https://huggingface.co/%s/tree/main/%s" % (repo_id, run_name)


@app.local_entrypoint()
def main(epochs: int = 3, limit: int = 0, batch_size: int = 32, name: str = "",
         hf_repo: str = "OnePunchMonk101010/laya-vision", no_push: bool = False):
    train_feats = cache_features.remote("train", limit)
    val_feats = cache_features.remote("validation", limit // 4 if limit else 0)
    run_name = name or ("smoke" if limit else "pilot_e%d" % epochs)
    cfg = {
        "run": run_name,
        "laya_id": "convaiinnovations/laya",
        "laya_subfolder": None,
        "vision_id": "google/siglip2-base-patch16-224",
        "dataset": "HuggingFaceM4/A-OKVQA",
        "train_feats": train_feats,
        "val_feats": val_feats,
        "out_dir": "/vol/runs/%s" % run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_text": 2e-5,
        "lr_new": 5e-4,
        "seed": 0,
        "limit": limit,
    }
    res = train.remote(cfg)
    os.makedirs("runs", exist_ok=True)
    with open("runs/%s.json" % run_name, "w") as f:
        json.dump(res, f, indent=2)
    keys = ["laya_text_only_zero_shot", "with_image", "shuffled_image", "latency", "train_minutes"]
    print(json.dumps({k: res.get(k) for k in keys}, indent=2))
    if not no_push:
        print("pushed (private):", push.remote(run_name, hf_repo))
