import random

import pytest
import torch
from laya.common import build_sequence
from transformers import AutoTokenizer

from laya_vision.data import row_items
from laya_vision.model import VisionDecisionModel
from laya_vision.sequence import build_vision_sequence, collate


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


Q = {"t": "choice", "ins": "What color is the bus?", "crit": {"red": None, "blue": None, "green": None}}
ROW = {"question_id": "q1", "question": "What is the man holding?", "choices": ["bat", "cup", "kite", "bat"], "correct_choice_idx": 2}


def test_no_image_matches_laya(tok):
    ids, markers, _ = build_vision_sequence(tok, Q, 0)
    ref_ids, ref_markers = build_sequence(tok, "", Q, 511)
    assert ids == ref_ids and markers == ref_markers


def test_image_slots_follow_state(tok):
    ids, markers, start = build_vision_sequence(tok, Q, 49, text_state="caption")
    assert len(markers) == 3 and all(ids[m] == tok.mask_token_id for m in markers)
    assert max(markers) < start
    assert ids[start - 1] == tok.sep_token_id and ids[-1] == tok.sep_token_id
    assert ids[start : start + 49] == [tok.pad_token_id] * 49 and len(ids) == start + 50


def test_row_items_labels(tok):
    train = row_items(ROW, 0, tok, 49, True, random.Random(0), 512, 192)
    ch = train[0]
    assert ch["kind"] == "choice" and 0 <= ch["label"] < 4 and len(ch["markers"]) == 4
    ev = row_items(ROW, 0, tok, 49, False, random.Random(0), 512, 192)
    assert ev[0]["label"] == 2  # eval keeps the original option order
    assert [it["label"] for it in ev[1:]] == [0, 0, 1, 0]  # duplicate "bat" stays a wrong answer


def test_forward_writes_image_tokens(tok):
    from transformers import AutoConfig, AutoModel
    from laya.common import DecisionModel

    cfg = AutoConfig.from_pretrained("answerdotai/ModernBERT-base", num_hidden_layers=2, hidden_size=64,
                                     intermediate_size=128, num_attention_heads=2, global_attn_every_n_layers=1)
    cfg.reference_compile = False
    enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
    m = VisionDecisionModel(DecisionModel(enc, head_layers=1), vision_dim=32, n_img=4).eval()

    items = row_items(ROW, 0, tok, 4, False, random.Random(0), 512, 192)
    feats = torch.randn(1, 4, 32)
    b = collate(items, tok.pad_token_id, feats)
    args = [b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    with torch.no_grad():
        a = m(*args, img_start=b["img_start"], img_feats=b["img_feats"])
        c = m(*args, img_start=b["img_start"], img_feats=torch.randn_like(b["img_feats"]))
    assert a.shape == (len(items), 4)
    assert torch.isinf(a).logical_not().all()
    assert not torch.allclose(a[:, :2], c[:, :2]), "image features must affect the logits"
