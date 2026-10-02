"""Dataset building and a leak-resistant split.

INCLUDE filenames carry no signer ID. Takes of one sign are recorded back-to-back (consecutive MVI numbers),
so we group consecutive numbers into 'sessions' and never split a session across train/val/test.
This is a PROXY for signer independence, not a proof of it. Report it as such.
"""
import re
import numpy as np
from .features import prepare_sequence

MVI = re.compile(r"MVI_(\d+)")


def mvi_number(name, default=-1):
    m = MVI.search(name)
    return int(m.group(1)) if m else default


def session_groups(mvis, max_gap=2):
    """Per class: sort by MVI number; start a new group when the gap exceeds max_gap. Returns group id per item."""
    order = np.argsort(mvis, kind="stable")
    gid = np.zeros(len(mvis), int)
    g, prev = 0, None
    for i in order:
        if prev is not None and mvis[i] - prev > max_gap:
            g += 1
        gid[i] = g
        prev = mvis[i]
    return gid


def group_split(labels, mvis, seed=0, frac_test=0.15, frac_val=0.15):
    """Returns array of 'train'/'val'/'test' per sample, and a dict of classes with too few sessions."""
    labels, mvis = np.asarray(labels), np.asarray(mvis)
    split = np.array(["train"] * len(labels), dtype=object)
    rng = np.random.RandomState(seed)
    low = {}
    for c in np.unique(labels):
        ix = np.flatnonzero(labels == c)
        gid = session_groups(mvis[ix])
        groups = list(np.unique(gid))
        if len(groups) < 3:
            low[str(c)] = len(groups)
            continue
        rng.shuffle(groups)
        n = len(ix)
        counts = {g: int((gid == g).sum()) for g in groups}
        want_t, want_v = max(1, round(n * frac_test)), max(1, round(n * frac_val))
        got_t = got_v = 0
        for g in groups[:-1]:  # always keep at least one group for training
            if got_t < want_t:
                split[ix[gid == g]] = "test"; got_t += counts[g]
            elif got_v < want_v:
                split[ix[gid == g]] = "val"; got_v += counts[g]
    return split, low


def build_arrays(paths, classes):
    """Load raw per-frame npy -> prepared (48,225) arrays."""
    cidx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for p in paths:
        c = p.replace("\\", "/").split("/")[-2]
        X.append(prepare_sequence(np.load(p).astype(np.float32)))
        y.append(cidx[c])
    return np.stack(X), np.array(y)
