import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from .model import ISLBiLSTM


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def augment(x):
    x = x * (1 + 0.1 * (torch.rand(x.size(0), 1, 1, device=x.device) - 0.5) * 2)       # global scale
    x = x + 0.01 * torch.randn_like(x)                                                    # jitter
    for sl in (slice(0, 63), slice(63, 126)):                                             # simulate dropped hand frames
        drop = (torch.rand(x.size(0), x.size(1), 1, device=x.device) < 0.05).float()
        x = torch.cat([x[..., :sl.start], x[..., sl] * (1 - drop), x[..., sl.stop:]], -1)
    return x


def fit(Xtr, ytr, Xva, yva, n_classes, device="cpu", epochs=80, patience=12, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ISLBiLSTM(n_classes).to(device)
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    model.mu.copy_(torch.tensor(flat.mean(0))); model.sd.copy_(torch.tensor(flat.std(0) + 1e-3))
    cnt = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor(cnt.sum() / (n_classes * np.maximum(cnt, 1)), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    Xt, yt = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device)
    Xv, yv = torch.tensor(Xva, device=device), torch.tensor(yva, device=device)
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=device)
        for batch_index, i in enumerate(range(0, len(Xt), 32)):
            b = perm[i:i + 32]
            loss = F.cross_entropy(model(augment(Xt[b])), yt[b], weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sched.step(ep + batch_index / max(1, len(Xt) // 32))
        model.eval()
        with torch.no_grad():
            lo = model(Xv); vl = F.cross_entropy(lo, yv).item(); va = (lo.argmax(1) == yv).float().mean().item()
        if ep % 5 == 0: log(f"ep {ep:3d} val_loss {vl:.3f} val_acc {va:.3f}")
        if va > best_acc: best_acc, best_state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def logits_of(model, X, device="cpu"):
    model.eval(); return model(torch.tensor(X, device=device)).cpu()


def fit_temperature(val_logits, yv):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    y = torch.tensor(yv)
    def closure():
        opt.zero_grad(); l = F.cross_entropy(val_logits / logT.exp(), y); l.backward(); return l
    opt.step(closure)
    return float(logT.exp())


def evaluate(logits, y, T=1.0):
    probs = F.softmax(logits / T, 1).numpy(); pred = probs.argmax(1)
    top3 = np.argsort(-probs, 1)[:, :3]
    return {"n": int(len(y)), "top1": float((pred == y).mean()),
            "top3": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
            "macro_f1": float(f1_score(y, pred, average="macro")), "ece": ece(probs, y)}


def save_checkpoint(path, model, classes, T, meta):
    torch.save({"state": model.state_dict(), "classes": classes, "temperature": T, "meta": meta}, path)
