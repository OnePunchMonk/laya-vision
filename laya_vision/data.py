"""A-OKVQA as Laya typed questions.

Each row gives one `choice` item (four options) and, for training, one `noul` item asking whether a
single proposed answer is correct (50% correct). For eval, every option becomes a `noul` item, so
noul accuracy is reported on a 1:3 true:false split alongside balanced accuracy.
"""
import random
from typing import Dict, Iterable, List

from laya.common import QTYPES

from .sequence import build_vision_sequence


def _unique(choices: List[str]) -> List[str]:
    seen, out = {}, []
    for c in choices:
        c = c.strip()
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else "%s (%d)" % (c, seen[c]))
    return out


def noul_question(question: str, answer: str) -> Dict:
    return {"t": "noul", "ins": 'Question about the image: %s Proposed answer: "%s". Is the proposed answer correct?' % (question, answer), "crit": None}


def row_items(row: Dict, img_row: int, tok, n_img: int, train: bool, rng: random.Random, max_len: int, head_max_len: int) -> List[Dict]:
    choices, gold = _unique(row["choices"]), int(row["correct_choice_idx"])
    items = []

    order = list(range(len(choices)))
    if train:
        rng.shuffle(order)
    q = {"t": "choice", "ins": row["question"], "crit": {c: None for c in choices}}
    ids, markers, start = build_vision_sequence(tok, q, n_img, max_len=max_len, head_max_len=head_max_len, option_order=order)
    items.append({"ids": ids, "markers": markers, "img_start": start, "img_row": img_row, "qtype": QTYPES["choice"],
                  "label": order.index(gold), "kind": "choice", "qid": row["question_id"]})

    if train:
        cands = [gold if rng.random() < 0.5 else rng.choice([i for i in range(len(choices)) if i != gold])]
    else:
        cands = list(range(len(choices)))
    for c in cands:
        ids, markers, start = build_vision_sequence(tok, noul_question(row["question"], choices[c]), n_img, max_len=max_len, head_max_len=head_max_len)
        items.append({"ids": ids, "markers": markers, "img_start": start, "img_row": img_row, "qtype": QTYPES["noul"],
                      "label": int(c == gold), "kind": "noul", "qid": row["question_id"], "opt": c})
    return items


def build_items(rows: Iterable[Dict], tok, n_img: int, train: bool, seed: int = 0, max_len: int = 512, head_max_len: int = 192) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(rows):
        out.extend(row_items(row, i, tok, n_img, train, rng, max_len, head_max_len))
    return out
