"""Laya decision model with an image branch.

A frozen SigLIP2 vision tower produces patch features (pooled offline, see features.py). A small
projector maps them into the text encoder's embedding space, and they fill the image slots of the
sequence. Everything after the encoder (type embedding, head layers, marker scorer) is Laya's own.
"""
from typing import Optional

import torch
import torch.nn as nn
from laya.common import DecisionModel


class VisionDecisionModel(nn.Module):
    def __init__(self, decision: DecisionModel, vision_dim: int, n_img: int):
        super().__init__()
        self.decision = decision
        d = decision.encoder.config.hidden_size
        self.n_img = n_img
        self.proj = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, d), nn.GELU(), nn.Linear(d, d))
        self.img_pos = nn.Parameter(torch.zeros(n_img, d))

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        img_start: Optional[torch.Tensor] = None,
        img_feats: Optional[torch.Tensor] = None,
    ):
        dm = self.decision
        emb = dm.encoder.get_input_embeddings()(input_ids)
        if img_feats is not None:
            img = self.proj(img_feats.to(emb.dtype)) + self.img_pos
            idx = img_start[:, None] + torch.arange(img.size(1), device=img.device)[None]
            emb = emb.scatter(1, idx[..., None].expand(-1, -1, emb.size(-1)), img.to(emb.dtype))
        h = dm.encoder(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        h = h + dm.type_emb(qtype)[:, None, :]
        if dm.head is not None:
            pad = ~attention_mask.bool()
            for layer in dm.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = dm.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def load_laya(model_id: str = "convaiinnovations/laya", subfolder: Optional[str] = None, device: str = "cuda"):
    """Load a released Laya checkpoint through laya's own loader. Returns (decision_model, tokenizer, cfg)."""
    from laya import Agent

    agent = Agent(model_id, device=device, subfolder=subfolder)
    return agent.model, agent.tok, agent.cfg
