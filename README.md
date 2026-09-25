# laya-vision

Typed decisions over **images** in one forward pass, built on [Laya](https://github.com/NandhaKishorM/laya).

Laya answers typed questions (`choice`, `score`, `noul`) about text without generating tokens. This repo
asks whether the same thing works when the state is an image: same question format, same decision head,
same proper-scoring objective, with a frozen vision encoder feeding the text encoder.

Status: private experiment. Pilot numbers below; nothing here is a benchmark yet.

## How it works

```
[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] text_state [SEP] <img x 49> [SEP]
```

- **Text side:** the released `convaiinnovations/laya` checkpoint (ModernBERT-large, 421M), loaded with
  Laya's own loader. Its type embedding, head layers and marker scorer are reused as is.
- **Image side:** `google/siglip2-base-patch16-224`, frozen. 14x14 patch features are average-pooled to
  7x7 = 49 tokens, passed through a LayerNorm + 2-layer MLP projector with a learned position table, and
  written into the image slots of the sequence.
- **Objective:** negative Laya proper reward (log score + spherical score, plus RPS for `score`).
- **Features are cached** once per split since the vision tower is frozen, so training only runs the
  text encoder.

## Pilot task: A-OKVQA

Each row becomes one `choice` item (4 options, shuffled in training) and one `noul` item ("is this
proposed answer correct?", 50% true). For eval, each option becomes a `noul` item.

Reported on the A-OKVQA **validation** split (1,145 questions), never used for selection:

| condition | what it measures |
|---|---|
| `laya_text_only_zero_shot` | released Laya, question and options only. The text-prior floor |
| `with_image` | trained model, correct image |
| `shuffled_image` | trained model, each question gets another question's image. If this is close to `with_image`, the model is not using the image |

Plus `choice_via_noul_acc` (pick the option with highest p(true)), ECE, and end-to-end latency on raw
pixels (SigLIP2 + one decision forward) for 1 and 10 questions on one image.

## Results

See `RESULTS.md` (written after each Modal run).

## Run

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q tests            # CPU, tiny random encoder

modal run modal_app.py --limit 400 --epochs 1  # smoke, a few minutes on one L4
modal run modal_app.py                         # pilot, full train split, 3 epochs
```

Needs a Modal secret named `huggingface`. Outputs land on the `laya-vision` volume under `runs/<name>/`
and a local copy of `results.json` in `runs/`.

## Where this is going

1. **A fast compositional reward for diffusion RL.** Break a prompt into yes/no and counting questions
   (GenEval / DSG style) and score each generated image in one pass, instead of a multi-second VLM judge.
   Needs a `score` head for counts and a dataset of generated images with question labels (teacher: a
   Qwen2.5-VL-7B class model on Modal).
2. **Screenshot state for web agents.** "Is the login form visible?", "which of these elements should be
   clicked?". The same use case Jev targets on text.
3. **Multilingual.** Swap in `laya-multilingual` (mmBERT-base) for questions in 100+ languages.
4. **Calibration.** Fit per-bucket temperatures as Laya does.
