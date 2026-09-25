"""Laya as the decision layer of a VLM judge: run and compare backends on Modal.

    modal run modal_judge.py --backend letters --smoke        # tiny subsets, checks wiring
    modal run --detach modal_judge.py --backend letters       # VLM-as-judge baseline, zero-shot
    modal run --detach modal_judge.py --backend caption       # VLM caption -> released Laya, zero-shot
    modal run --detach modal_judge.py --backend vlm_head      # train LoRA + Laya head, then evaluate

Results: /vol/judge/<run>/results.json on the `laya-vision` volume, copied to runs/judge_<run>.json.
The trained vlm_head (adapter + head + results) is pushed to the private HF repo.
"""
import json
import os

import modal

app = modal.App("laya-judge")
vol = modal.Volume.from_name("laya-vision", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "transformers==4.51.3", "peft==0.15.2", "accelerate>=1.2",
        "datasets>=2.18", "safetensors>=0.4", "laya>=0.3.4", "pillow", "numpy", "huggingface_hub",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya_vision")
)
common = dict(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface")])

CFG = {
    "vlm_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "laya_id": "convaiinnovations/laya",
    # eval
    "val_rows": 0,            # A-OKVQA validation rows, 0 = all 1,145
    "test_prompts": 0,        # GenAI-Bench test prompts, 0 = all 400 (x6 images)
    # vlm_head training
    "aokvqa_rows": 4000,
    "swap_pairs": 0.5,
    "genai_prompts": 0,       # GenAI-Bench train prompts, 0 = all 1,200
    "batch_size": 8,
    "eval_batch_size": 32,
    "lr_lora": 1e-4,
    "lr_head": 1e-3,
    "lora_r": 16,
    "max_steps": 0,
    "seed": 0,
}
SMOKE = {"val_rows": 24, "test_prompts": 6, "aokvqa_rows": 48, "genai_prompts": 6, "max_steps": 12}


def eval_tasks(cfg):
    from laya_vision.judge.tasks import aokvqa, genai_bench

    tasks = {}
    for e in aokvqa("validation", rows=cfg["val_rows"], noul="all") + genai_bench("test", prompts=cfg["test_prompts"]):
        tasks.setdefault(e.task, []).append(e)
    return tasks


@app.function(gpu="A100", timeout=6 * 3600, **common)
def run(backend: str, cfg: dict, run_name: str) -> dict:
    from laya_vision.judge.evaluate import evaluate, latency

    out_dir = "/vol/judge/%s" % run_name
    os.makedirs(out_dir, exist_ok=True)
    tasks = eval_tasks(cfg)
    print({k: len(v) for k, v in tasks.items()}, flush=True)

    if backend == "letters":
        from laya_vision.judge.backends import LetterJudge
        judge = LetterJudge(cfg["vlm_id"], batch_size=cfg["eval_batch_size"])
    elif backend == "caption":
        from laya_vision.judge.backends import CaptionLayaJudge
        judge = CaptionLayaJudge(cfg["vlm_id"], cfg["laya_id"], cache_path="/vol/judge/captions_%s.json" % cfg["vlm_id"].split("/")[-1])
    elif backend == "vlm_head":
        from laya_vision.judge.train import train
        judge = train({**cfg, "out_dir": out_dir})
    else:
        raise ValueError(backend)

    res = {"backend": backend, "config": cfg, **evaluate(judge, tasks)}
    res["latency_aokvqa_5q"] = latency(judge, tasks["aokvqa_choice"] + tasks["aokvqa_noul"])
    res["latency_genai_1q"] = latency(judge, tasks["genai_score"])
    if hasattr(judge, "train_minutes"):
        res["train_minutes"] = judge.train_minutes
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    vol.commit()
    if backend == "vlm_head" and cfg.get("push_repo"):
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(cfg["push_repo"], private=True, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=cfg["push_repo"], path_in_repo="judge/%s" % run_name, commit_message="Add judge run %s" % run_name)
        res["hf_url"] = "https://huggingface.co/%s/tree/main/judge/%s" % (cfg["push_repo"], run_name)
    return res


@app.local_entrypoint()
def main(backend: str = "letters", smoke: bool = False, name: str = "", no_push: bool = False,
         hf_repo: str = "OnePunchMonk101010/laya-vision"):
    cfg = {**CFG, **(SMOKE if smoke else {})}
    cfg["push_repo"] = None if (no_push or smoke) else hf_repo
    run_name = name or ("%s%s" % (backend, "_smoke" if smoke else ""))
    res = run.remote(backend, cfg, run_name)
    os.makedirs("runs", exist_ok=True)
    json.dump(res, open("runs/judge_%s.json" % run_name, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
