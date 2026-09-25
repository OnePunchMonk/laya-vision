"""Three ways to judge an image with Laya-typed questions.

letters     The usual VLM-as-judge, done in one pass: prompt the VLM with lettered options (or Yes/No,
            or 1..K) and softmax its next-token logits over those answer tokens. Baseline.
caption     The VLM writes one detailed description per image; released Laya answers every typed
            question about that text. No training. Laya replaces the VLM's text reasoning.
vlm_head    The VLM reads image + question + options, each option closed by a marker token. Laya's
            typed scorer reads the hidden state at each marker. No decoding. Trained with LoRA and
            Laya's proper-scoring objective.
"""
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from laya.common import proper_reward, render_options

from .core import Example, to_user_question

VLM_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MARK = "<|quad_end|>"  # unused Qwen special token, closes each option for vlm_head
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_vlm(model_id: str = VLM_ID, device: str = "cuda", max_pixels: int = 448 * 448, min_pixels: int = 128 * 128):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return proc, model.to(device).eval()


def chat(proc, text: str, generation_prompt: bool) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation_prompt)


_POOL = None


def _images(examples, image_fns):
    """Decode and resize in threads; on A100 the single-threaded PIL path was the bottleneck."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(8)
    return list(_POOL.map(lambda f: f(), image_fns or [e.image_fn for e in examples]))


def _chunks(n: int, bs: int):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


# ---------------------------------------------------------------- letters

class LetterJudge:
    name = "letters"

    def __init__(self, model_id: str = VLM_ID, device: str = "cuda", batch_size: int = 16, vlm=None):
        self.proc, self.model = vlm or load_vlm(model_id, device)
        self.proc.tokenizer.padding_side = "left"
        self.device, self.bs = device, batch_size
        enc = lambda s: self.proc.tokenizer.encode(s, add_special_tokens=False)[0]
        self.letter_ids = [enc(c) for c in LETTERS]
        self.yes_no = [enc("No"), enc("Yes")]
        self.digits = [enc(str(i)) for i in range(1, 10)]

    def prompt(self, q: Dict):
        opts, t = render_options(q), q["t"]
        if t == "choice":
            body = "%s\nOptions:\n%s\nAnswer with the option's letter only." % (q["ins"], "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(opts)))
            return body, self.letter_ids[: len(opts)]
        if t == "noul":
            return "%s\nAnswer Yes or No." % q["ins"], self.yes_no
        crit = q["crit"]
        body = "%s\n%s\nAnswer with a single number from 1 to %d." % (q["ins"], "\n".join("%d: %s" % (i + 1, c) for i, c in enumerate(crit)), len(crit))
        return body, self.digits[: len(crit)]

    @torch.no_grad()
    def predict(self, examples: List[Example], image_fns=None, image_keys=None) -> List[np.ndarray]:
        out = []
        for sl in _chunks(len(examples), self.bs):
            exs = examples[sl]
            imgs = _images(exs, image_fns[sl] if image_fns else None)
            prompts, cands = zip(*[self.prompt(e.q) for e in exs])
            b = self.proc(text=[chat(self.proc, p, True) for p in prompts], images=imgs, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**b).logits[:, -1].float()
            for r, c in enumerate(cands):
                out.append(torch.softmax(logits[r, c], -1).cpu().numpy())
        return out


# ---------------------------------------------------------------- caption -> Laya

CAPTION_PROMPT = ("Describe this image in detail for someone who cannot see it: every object, how many of each, "
                  "colors, sizes, positions relative to each other, any text, and anything unusual or malformed.")


class CaptionLayaJudge:
    name = "caption"

    def __init__(self, model_id: str = VLM_ID, laya_id: str = "convaiinnovations/laya", device: str = "cuda",
                 cache_path: Optional[str] = None, batch_size: int = 24, max_new_tokens: int = 192, vlm=None):
        from laya import Agent

        self._vlm_args = (model_id, device)
        self._vlm = vlm
        self.agent = Agent(laya_id, device=device)
        self.cache_path, self.bs, self.max_new = cache_path, batch_size, max_new_tokens
        self.captions: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            self.captions = json.load(open(cache_path))

    @torch.no_grad()
    def caption(self, keys: List[str], fns):
        todo = {}
        for k, f in zip(keys, fns):
            if k not in self.captions:
                todo.setdefault(k, f)
        if not todo:
            return
        proc, model = self._vlm or load_vlm(*self._vlm_args)
        self._vlm = (proc, model)
        proc.tokenizer.padding_side = "left"
        keys = list(todo)
        t0 = time.time()
        for sl in _chunks(len(keys), self.bs):
            ks = keys[sl]
            b = proc(text=[chat(proc, CAPTION_PROMPT, True)] * len(ks), images=_images(None, [todo[k] for k in ks]), return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**b, max_new_tokens=self.max_new, do_sample=False)
            for k, g in zip(ks, proc.batch_decode(gen[:, b["input_ids"].shape[1]:], skip_special_tokens=True)):
                self.captions[k] = g.strip()
            if (sl.start // self.bs) % 10 == 0:
                print("captions %d/%d (%.0fs)" % (sl.stop, len(keys), time.time() - t0), flush=True)
        if self.cache_path and not any(k.startswith("__") for k in keys):
            json.dump(self.captions, open(self.cache_path, "w"))

    def predict(self, examples: List[Example], image_fns=None, image_keys=None) -> List[np.ndarray]:
        keys = image_keys or [e.image_key for e in examples]
        self.caption(keys, image_fns or [e.image_fn for e in examples])
        out: List[Optional[np.ndarray]] = [None] * len(examples)
        by_state: Dict[str, List[int]] = {}
        for i, k in enumerate(keys):
            by_state.setdefault(k, []).append(i)
        for k, idx in by_state.items():
            res = self.agent.system_one(self.captions[k], {str(i): to_user_question(examples[i].q) for i in idx})["answers"]
            for i in idx:
                a, q = res[str(i)], examples[i].q
                if q["t"] == "noul":
                    out[i] = np.array([1 - a["noul"], a["noul"]])
                elif q["t"] == "choice":
                    out[i] = np.array([a["probabilities"][c] for c in q["crit"]])
                else:
                    out[i] = np.array([a["probabilities"][str(j)] for j in range(len(q["crit"]))])
                out[i] = out[i] / out[i].sum()
        return out


# ---------------------------------------------------------------- VLM hidden states + Laya typed head

class TypedHead(nn.Module):
    """Laya's decision readout: type embedding + MLP scorer over option-marker hidden states."""

    def __init__(self, d: int):
        super().__init__()
        self.type_emb = nn.Embedding(3, d)
        nn.init.zeros_(self.type_emb.weight)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, h_markers, qtype, marker_mask):
        z = self.scorer(h_markers.float() + self.type_emb(qtype)[:, None, :]).squeeze(-1)
        return z.masked_fill(~marker_mask, -1e4)


class VLMHeadJudge(nn.Module):
    name = "vlm_head"

    def __init__(self, model_id: str = VLM_ID, device: str = "cuda", lora_r: int = 16, vlm=None, batch_size: int = 16):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.proc, base = vlm or load_vlm(model_id, device)
        self.proc.tokenizer.padding_side = "right"
        self.mark_id = self.proc.tokenizer.convert_tokens_to_ids(MARK)
        d = getattr(base.config, "hidden_size", None) or base.config.text_config.hidden_size
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                         target_modules=r".*model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
        self.vlm = get_peft_model(base, cfg)
        self.head = TypedHead(d).to(device)
        self.device, self.bs = device, batch_size

    def prompt(self, q: Dict, order: Optional[List[int]] = None) -> str:
        opts = render_options(q)
        order = order or list(range(len(opts)))
        return "%s question: %s\nOptions:%s" % (q["t"], q["ins"], "".join("\n- %s%s" % (opts[i], MARK) for i in order))

    def forward(self, examples: List[Example], images, orders=None):
        orders = orders or [None] * len(examples)
        texts = [chat(self.proc, self.prompt(e.q, o), False) for e, o in zip(examples, orders)]
        b = self.proc(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        h = self.vlm(**b, output_hidden_states=True).hidden_states[-1]
        pos = [(row == self.mark_id).nonzero(as_tuple=True)[0] for row in b["input_ids"]]
        kmax = max(len(p) for p in pos)
        mpos = torch.zeros(len(pos), kmax, dtype=torch.long, device=h.device)
        mmask = torch.zeros(len(pos), kmax, dtype=torch.bool, device=h.device)
        for r, p in enumerate(pos):
            assert len(p) == examples[r].k, "marker count %d != options %d" % (len(p), examples[r].k)
            mpos[r, : len(p)], mmask[r, : len(p)] = p, True
        hm = torch.gather(h, 1, mpos[..., None].expand(-1, -1, h.size(-1)))
        qtype = torch.tensor([e.qtype for e in examples], device=h.device)
        return self.head(hm, qtype, mmask), mmask

    @torch.no_grad()
    def predict(self, examples: List[Example], image_fns=None, image_keys=None) -> List[np.ndarray]:
        self.eval()
        out = []
        for sl in _chunks(len(examples), self.bs):
            exs = examples[sl]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, m = self(exs, _images(exs, image_fns[sl] if image_fns else None))
            p = torch.softmax(z.float(), -1).cpu().numpy()
            out.extend(p[r, : e.k] for r, e in enumerate(exs))
        return out

    def loss(self, examples: List[Example], images, rng: np.random.Generator):
        """Options shuffled per example; targets permuted to match. Negative Laya proper reward."""
        orders = [list(rng.permutation(e.k)) if e.q["t"] == "choice" else None for e in examples]
        z, m = self(examples, images, orders)
        tgt = torch.zeros_like(z)
        for r, (e, o) in enumerate(zip(examples, orders)):
            g = e.gold if o is None else [e.gold[i] for i in o]
            tgt[r, : e.k] = torch.tensor(g, device=z.device)
        qtype = torch.tensor([e.qtype for e in examples], device=z.device)
        return -proper_reward(torch.softmax(z.float(), -1), tgt, qtype, m.float()).mean()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.vlm.save_pretrained(path)
        torch.save(self.head.state_dict(), os.path.join(path, "typed_head.pt"))
