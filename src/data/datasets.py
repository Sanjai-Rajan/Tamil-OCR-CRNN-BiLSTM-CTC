"""
Datasets for training / evaluation.

Every source yields (uint8 grayscale image, NFC label, source name).
  * ManifestDataset - real images listed in a JSONL manifest {"image": path, "label": text}
  * SynthSource     - on-the-fly synthetic printed words (src/synth)
  * MixtureDataset  - draws from several sources with fixed proportions each epoch
  * WidthBucketSampler - optional batch sampler that groups words of similar width
collate() applies the shared preprocessing (src/preprocess.py) and pads the batch.
"""
import json
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src import charset as cs
from src import preprocess as pp

PAD_W_MULTIPLE = 32   # batch width is rounded up to this; true widths are passed separately, so the LSTM ignores it


def read_image(path: str) -> np.ndarray | None:
    # np.fromfile + imdecode handles non-ASCII Windows paths (cv2.imread does not)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:  # e.g. some TIFF variants
            from PIL import Image
            img = np.asarray(Image.open(path).convert("L"))
        return img
    except Exception:
        return None


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, name: str | None = None, augment=None, max_items: int | None = None,
                 seed: int = 0, where: dict | None = None):
        """where: keep only entries whose fields match, e.g. {"style": "old"} for the synthetic val set."""
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.name = name or self.manifest.stem
        self.augment = augment
        self.items, self.dropped = [], 0
        with open(self.manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if where and any(e.get(k) != v for k, v in where.items()):
                    continue
                label = cs.normalize(e["label"])
                if not cs.is_encodable(label):
                    self.dropped += 1
                    continue
                p = Path(e["image"])
                self.items.append((str(p if p.is_absolute() else self.root / p), label))
        if max_items and len(self.items) > max_items:       # deterministic subset (for quick validation)
            self.items = random.Random(seed).sample(self.items, max_items)
        self.epoch = 0
        self._name_id = zlib.crc32(self.name.encode())   # stable across processes (str hash is not)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    # -- width estimates for WidthBucketSampler (image headers only; the images themselves are not decoded)
    def aspects(self) -> np.ndarray:
        if getattr(self, "_aspects", None) is None:
            from PIL import Image

            def one(item):
                try:
                    with Image.open(item[0]) as im:
                        w, h = im.size
                    return w / max(1, h)
                except Exception:
                    return 4.0
            with ThreadPoolExecutor(16) as ex:
                self._aspects = np.fromiter(ex.map(one, self.items, chunksize=256), np.float32, len(self.items))
        return self._aspects

    def est_width(self, idx):
        return est_input_width(self.aspects()[idx])

    def get(self, idx, rng: random.Random | None = None):
        path, label = self.items[idx]
        img = read_image(path)
        if img is None:
            img = np.full((32, 64), 255, np.uint8)
            label = ""
        if self.augment is not None and rng is not None:
            img = self.augment(img, rng)
        return img, label, self.name

    def __getitem__(self, idx):
        rng = random.Random(hash((self.epoch, idx, self._name_id)) & 0x7FFFFFFF) if self.augment else None
        return self.get(idx, rng)


class SynthSource:
    """Adapter so the synthetic generator looks like a dataset of `length` items."""

    def __init__(self, generator, length: int, name: str, seed_base: int = 0, fixed: bool = False):
        self.gen, self.length, self.name = generator, length, name
        self.seed_base, self.fixed = seed_base, fixed

    def __len__(self):
        return self.length

    def est_width(self, idx):
        return None                      # unknown until generated; the sampler uses the median

    def get(self, idx, rng=None, epoch=0):
        seed = self.seed_base + idx if self.fixed else (hash((self.seed_base, epoch, idx)) & 0x7FFFFFFF)
        if not self.fixed:
            # Seed leakage guard against frozen validation set (900000000..900003000, 950000000..950003000)
            if (900000000 <= seed <= 900003000) or (950000000 <= seed <= 950003000):
                seed ^= 0x12345678
        img, text, _ = self.gen.sample(seed)
        return img, text, self.name


class MixtureDataset(Dataset):
    """Each epoch has `epoch_size` samples; sample i is drawn from source k with probability w_k."""

    def __init__(self, sources: list, weights: list[float], epoch_size: int, seed: int = 0):
        assert len(sources) == len(weights) and sources
        tot = float(sum(weights))
        self.sources, self.weights = sources, [w / tot for w in weights]
        self.cum = np.cumsum(self.weights)
        self.epoch_size, self.seed, self.epoch = epoch_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.epoch_size

    def locate(self, idx):
        """Which source k and item j sample idx of the current epoch is (same draws as __getitem__)."""
        rng = random.Random(hash((self.seed, self.epoch, idx)) & 0x7FFFFFFF)
        k = int(np.searchsorted(self.cum, rng.random(), side="right"))
        k = min(k, len(self.sources) - 1)
        j = rng.randrange(len(self.sources[k]))
        return k, j, rng

    def est_width(self, idx):
        k, j, _ = self.locate(idx)
        return self.sources[k].est_width(j)

    def __getitem__(self, idx):
        k, j, rng = self.locate(idx)
        src = self.sources[k]
        if isinstance(src, SynthSource):
            return src.get(j, rng, self.epoch)
        return src.get(j, rng)


def est_input_width(aspect: float) -> int:
    """Model input width for an image of this aspect ratio (before augmentation, which changes it by ~+-25%)."""
    return int(np.clip(round(aspect * pp.TARGET_H), pp.MIN_W, pp.MAX_W))


class WidthBucketSampler:
    """Batches of words with similar width.

    Every image in a batch is padded to the widest one, and the CNN computes (and stores activations for)
    the padding too. With random batches one long word makes 63 short ones pay its width. Here each pool of
    `pool_batches` random batches is sorted by estimated width and cut into batches, and the batch order is
    shuffled again, so every sample of the epoch is still used exactly once and in random batch order.
    `max_columns` caps (images x padded width): batches of very long words get fewer images, bounding memory.
    Deterministic for a given (seed, epoch).
    """

    def __init__(self, dataset, batch_size: int, pool_batches: int = 50, max_columns: int | None = None,
                 shuffle: bool = True, seed: int = 0, unknown_width_estimate: int | None = None):
        self.ds, self.bs, self.pool = dataset, batch_size, pool_batches
        self.max_columns, self.shuffle, self.seed = max_columns, shuffle, seed
        self.unknown_width_estimate = unknown_width_estimate
        self.epoch, self._cache = 0, {}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _widths(self):
        if hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(self.epoch)
        w = [self.ds.est_width(i) for i in range(len(self.ds))]
        known = [x for x in w if x is not None]
        med = int(np.median(known)) if known else 256
        fallback = self.unknown_width_estimate if self.unknown_width_estimate is not None else med
        return np.array([fallback if x is None else x for x in w], np.int32)

    def batches(self, epoch=None):
        e = self.epoch if epoch is None else epoch
        if e in self._cache:
            return self._cache[e]
        old = self.epoch; self.epoch = e
        w = self._widths()
        self.epoch = old
        if hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(old)
        n = len(w)
        rng = random.Random(hash((self.seed, e, 7919)) & 0x7FFFFFFF)   # ints only: str hashes differ per process
        order = list(range(n))
        if self.shuffle:
            rng.shuffle(order)
        order = np.array(order, np.int64)
        out, P = [], self.pool * self.bs if self.shuffle else n
        for s in range(0, n, P):
            pool = order[s:s + P]
            pool = pool[np.argsort(w[pool], kind="stable")]
            cur, cur_w = [], 0
            for i in pool:
                wi = -(-int(w[i]) // PAD_W_MULTIPLE) * PAD_W_MULTIPLE
                fits = self.max_columns is None or (len(cur) + 1) * max(cur_w, wi) <= self.max_columns
                if cur and (len(cur) >= self.bs or not fits):
                    out.append(cur); cur, cur_w = [], 0
                cur.append(int(i)); cur_w = max(cur_w, wi)
            if cur:
                out.append(cur)
        if self.shuffle:
            rng.shuffle(out)
        self._cache = {e: out}
        return out

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())


def collate(batch):
    """Preprocess, pad to the widest image, and pack CTC targets."""
    tensors, pads, texts, sources = [], [], [], []
    for img, text, src in batch:
        if not text:                       # unreadable file: skip
            continue
        g = pp.resize_to_height(pp.to_gray(img))
        tensors.append(pp.to_tensor(g))
        pads.append((pp.background_value(g) / 255.0 - 0.5) / 0.5)
        texts.append(text)
        sources.append(src)
    if not tensors:
        return None
    H = tensors[0].shape[1]
    W = max(t.shape[2] for t in tensors)
    W = -(-W // PAD_W_MULTIPLE) * PAD_W_MULTIPLE      # few distinct batch shapes -> cheap cuDNN/allocator reuse
    images = torch.empty(len(tensors), 1, H, W)
    for i, (t, pv) in enumerate(zip(tensors, pads)):
        images[i].fill_(pv)
        images[i, :, :, : t.shape[2]] = t
    widths = torch.tensor([t.shape[2] for t in tensors], dtype=torch.long)
    enc = [cs.encode(t) for t in texts]
    targets = torch.tensor([i for e in enc for i in e], dtype=torch.long)
    target_lengths = torch.tensor([len(e) for e in enc], dtype=torch.long)
    return {"images": images, "widths": widths, "targets": targets,
            "target_lengths": target_lengths, "texts": texts, "sources": sources}
# Experiment 002-A: append this class verbatim to the END of src/data/datasets.py.
# It is purely additive - nothing existing is edited, so your three safety fixes stay intact.
# It uses only the unchanged APIs: MixtureDataset.locate(), <source>.est_width(), PAD_W_MULTIPLE.


class PairedSourceBatchSampler:
    """Batches that always contain every source, in the configured proportions (Experiment 002-A).

    WidthBucketSampler sorts each pool by estimated width. Synthetic samples all carry the same constant
    estimate, so they end up contiguous and the batches become nearly single-source: Experiment 001
    measured 58% synthetic-only, 40% handwriting-only, 2% mixed. This sampler instead splits the epoch's
    indices by source and takes a fixed share of each batch from each source.

      * a source whose est_width() is a real per-item value (the manifest sources) is sorted by width, so
        batches stay width-homogeneous within that source;
      * a source whose est_width() is None (the synthetic generator) is shuffled, because its estimate is a
        single constant and ordering it carries no information.

    The max_columns rule is exactly the one WidthBucketSampler applies - (images x padded width) must not
    exceed max_columns, judged on the same estimates - and when it binds, every source's share is scaled by
    the same factor so the mixture ratio inside the batch is preserved. Every index of the epoch is used
    exactly once. Deterministic for a given (seed, epoch).
    """

    def __init__(self, dataset, batch_size: int, weights=None, max_columns: int | None = None,
                 unknown_width: int | None = None, shuffle: bool = True, seed: int = 0):
        self.ds, self.bs = dataset, int(batch_size)
        self.weights = list(weights) if weights is not None else list(getattr(dataset, "weights", [1.0]))
        self.max_columns, self.shuffle, self.seed = max_columns, shuffle, seed
        self.unknown_width = unknown_width
        self.epoch, self._cache, self._stats = 0, {}, {}

    def set_epoch(self, epoch):
        self.epoch = epoch

    # -- how many indices each source contributes to a full batch (sums to batch_size)
    def _quota(self):
        tot = float(sum(self.weights)) or 1.0
        raw = [self.bs * w / tot for w in self.weights]
        q = [int(x) for x in raw]
        for k in sorted(range(len(raw)), key=lambda i: raw[i] - q[i], reverse=True):
            if sum(q) >= self.bs:
                break
            q[k] += 1
        return q

    def _widths_and_groups(self, e):
        ds = self.ds
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(e)
        n = len(ds)
        w = np.empty(n, np.int32)
        groups = [[] for _ in self.weights]
        unknown = [False] * len(self.weights)
        for i in range(n):
            k, j, _ = ds.locate(i)
            est = ds.sources[k].est_width(j)
            if est is None:
                unknown[k] = True
                est = self.unknown_width if self.unknown_width else 0
            w[i] = est
            groups[k].append(i)
        if not self.unknown_width:                      # same fallback as WidthBucketSampler
            known = [int(x) for x in w if x > 0]
            med = int(np.median(known)) if known else 256
            w[w == 0] = med
        return w, groups, unknown

    def batches(self, epoch=None):
        e = self.epoch if epoch is None else epoch
        if e in self._cache:
            return self._cache[e]
        old = self.epoch
        w, groups, unknown = self._widths_and_groups(e)
        if hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(old)
        rng = random.Random(hash((self.seed, e, 104729)) & 0x7FFFFFFF)   # ints only: reproducible anywhere
        order = []
        for k, idxs in enumerate(groups):
            if unknown[k]:
                idxs = list(idxs); rng.shuffle(idxs)                     # constant estimate: order is arbitrary
            else:
                idxs = sorted(idxs, key=lambda i: (int(w[i]), i))        # width-homogeneous within the source
            order.append(idxs)
        quota, ptr, out = self._quota(), [0] * len(groups), []
        tail = 0
        while True:
            take = [min(quota[k], len(order[k]) - ptr[k]) for k in range(len(groups))]
            if sum(take) == 0:
                break
            if self.max_columns:
                cand = [i for k in range(len(groups)) for i in order[k][ptr[k]:ptr[k] + take[k]]]
                wmax = -(-int(max(int(w[i]) for i in cand)) // PAD_W_MULTIPLE) * PAD_W_MULTIPLE
                allowed = max(1, self.max_columns // max(1, wmax))
                if sum(take) > allowed:                                  # scale every source by one factor
                    f = allowed / sum(take)
                    take = [max(1, int(t * f)) if t else 0 for t in take]
                    while sum(take) > allowed:
                        take[max(range(len(take)), key=lambda k: take[k])] -= 1
            batch = []
            for k in range(len(groups)):
                batch += order[k][ptr[k]:ptr[k] + take[k]]
                ptr[k] += take[k]
            tail += sum(1 for t in take if t == 0)                       # a source was exhausted
            out.append(batch)
        if self.shuffle:
            rng.shuffle(out)
        sizes = [len(b) for b in out]
        self._stats = {"batches": len(out), "mean_batch_size": float(np.mean(sizes)),
                       "min_batch_size": int(min(sizes)), "max_batch_size": int(max(sizes)),
                       "batches_missing_a_source": int(tail), "quota": self._quota()}
        self._cache = {e: out}
        return out

    def stats(self, epoch=None):
        self.batches(epoch)
        return dict(self._stats)

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())
