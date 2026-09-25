"""Precompute pooled SigLIP2 patch features. The vision tower is frozen, so this runs once per split."""
import torch
import torch.nn.functional as F

VISION_ID = "google/siglip2-base-patch16-224"


def load_vision(model_id: str = VISION_ID, device: str = "cuda", dtype=torch.float16):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_id)
    vm = AutoModel.from_pretrained(model_id, torch_dtype=dtype).vision_model.to(device).eval()
    return proc, vm


@torch.no_grad()
def encode_images(images, proc, vm, pool: int = 2) -> torch.Tensor:
    """PIL images -> [B, (G/pool)^2, D] patch features (14x14 grid pooled to 7x7 at pool=2)."""
    dev, dt = next(vm.parameters()).device, next(vm.parameters()).dtype
    px = proc(images=[im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"].to(dev, dt)
    h = vm(pixel_values=px).last_hidden_state
    b, n, d = h.shape
    g = int(n**0.5)
    h = h.transpose(1, 2).reshape(b, d, g, g)
    if pool > 1:
        h = F.avg_pool2d(h, pool)
    return h.flatten(2).transpose(1, 2)


def cache_split(split: str, out_path: str, dataset: str = "HuggingFaceM4/A-OKVQA", batch: int = 128, limit: int = 0):
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    proc, vm = load_vision()
    out = []
    for i in range(0, len(ds), batch):
        out.append(encode_images(ds[i : i + batch]["image"], proc, vm).cpu())
        if (i // batch) % 20 == 0:
            print("%s: %d/%d" % (split, i, len(ds)), flush=True)
    feats = torch.cat(out)
    torch.save({"feats": feats, "question_id": list(ds["question_id"])}, out_path)
    print("saved", out_path, tuple(feats.shape), flush=True)
    return tuple(feats.shape)
