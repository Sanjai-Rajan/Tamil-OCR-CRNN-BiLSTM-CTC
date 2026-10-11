"""
Recognition metrics, defined so they cannot be inflated by the dataset mix.

CER  = total edit distance / total reference characters   (micro-average: long words count more,
       unlike the old per-sample average where single letters dominated)
WER  = fraction of word images whose prediction is not an exact match (each sample is one word)
Also reported per source (handwritten / printed / synthetic ...) and per length bucket.
"""
from collections import defaultdict

import editdistance

BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 10_000)]


def bucket_name(n):
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "0"


class Scoreboard:
    def __init__(self):
        self.s = defaultdict(lambda: {"edits": 0, "chars": 0, "words": 0, "wrong": 0,
                                      "pred_chars": 0})

    def add(self, ref: str, hyp: str, source: str = "all"):
        e = editdistance.eval(ref, hyp)
        for key in ("all", f"src:{source}", f"len:{bucket_name(len(ref))}",
                    f"src:{source}|len:{bucket_name(len(ref))}"):
            d = self.s[key]
            d["edits"] += e; d["chars"] += len(ref); d["words"] += 1
            d["wrong"] += int(ref != hyp); d["pred_chars"] += len(hyp)

    def summary(self):
        out = {}
        for k, d in self.s.items():
            out[k] = {"n": d["words"],
                      "cer": d["edits"] / max(1, d["chars"]),
                      "wer": d["wrong"] / max(1, d["words"]),
                      "word_acc": 1 - d["wrong"] / max(1, d["words"]),
                      "len_ratio": d["pred_chars"] / max(1, d["chars"])}
        return out
