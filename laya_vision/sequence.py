"""Token layout for image questions.

Laya's text layout is  [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].
Laya-Vision keeps it and appends the image as a second state segment:

    [CLS] <type> instructions [SEP] [MASK] opt0 ... [SEP] text_state [SEP] <img_0 .. img_N-1> [SEP]

The image slots hold pad ids in `ids`; the model overwrites their embeddings with projected
vision features. Markers sit before the image, so their positions are unchanged from Laya.
"""
from typing import Dict, List, Optional

import torch
from laya.common import build_sequence, render_options


def build_vision_sequence(
    tok,
    q: Dict,
    n_img: int,
    text_state: str = "",
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
):
    """Return (ids, markers, img_start). With n_img == 0 this is plain text-only Laya."""
    ids, markers = build_sequence(
        tok, text_state, q, max_len=max_len - n_img - 1, head_max_len=head_max_len, option_order=option_order
    )
    if len(markers) != len(render_options(q)):
        raise ValueError("options exceed head_max_len=%d" % head_max_len)
    img_start = len(ids)
    if n_img:
        ids = ids + [tok.pad_token_id] * n_img + [tok.sep_token_id]
    return ids, markers, img_start


def collate(items: List[Dict], pad_id: int, feats: Optional[torch.Tensor] = None) -> Dict:
    """Pad a flat list of items. `feats[img_row]` supplies each item's image tokens."""
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if it.get("label", -1) >= 0:
            target[i, it["label"]] = 1.0
    out = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "target": target,
        "img_start": torch.tensor([it["img_start"] for it in items]),
    }
    if feats is not None:
        out["img_feats"] = feats[torch.tensor([it["img_row"] for it in items])]
    return out
