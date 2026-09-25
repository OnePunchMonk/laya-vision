"""Shared types for image judges.

Every judge answers Laya-typed questions (`choice`, `score`, `noul`) about an image and returns a
probability vector over `render_options(q)`, in that order. That single contract is what lets the
caption, letter and hidden-state backends be scored by one evaluator.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
from laya.common import QTYPES, render_options

ImageFn = Callable[[], "PIL.Image.Image"]  # noqa: F821


@dataclass
class Example:
    key: str                  # unique per example
    image_key: str            # examples that share an image share this
    image_fn: ImageFn         # lazy loader, images are large
    q: Dict                   # internal Laya question {"t", "ins", "crit"}
    gold: List[float]         # target distribution over render_options(q); one-hot or soft (human ratings)
    task: str = ""
    group: str = ""           # choice/noul: source question; score: prompt (for pairwise ranking)
    human: float = float("nan")  # score tasks: mean human rating on the 1..K scale
    meta: Dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(render_options(self.q))

    @property
    def qtype(self) -> int:
        return QTYPES[self.q["t"]]


class Judge(Protocol):
    name: str

    def predict(self, examples: List[Example], image_fns: Optional[List[ImageFn]] = None,
                image_keys: Optional[List[str]] = None) -> List[np.ndarray]:
        """Probabilities over render_options(ex.q) per example. `image_fns`/`image_keys` override the
        example's own image (used for the shuffled-image check)."""
        ...


def to_user_question(q: Dict) -> Dict:
    """Internal Laya question -> the public Laya API format."""
    crit = q.get("crit")
    return {"type": q["t"], "instructions": q["ins"], **({"criteria": crit} if crit is not None else {})}


def fit_image(img, max_side: int = 640):
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    return img
