import numpy as np, torch, torch.nn.functional as F
from .model import ISLBiLSTM
from .features import FEATURE_DIM


def _load_checkpoint(path, device):
    """Load checkpoint; retry with weights_only=False on UnpicklingError (older PyTorch ckpts)."""
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        # Covers both pickle.UnpicklingError and torch FutureWarning variants
        if "pickle" in str(type(e).__name__).lower() or "unpickling" in str(e).lower() or "weights_only" in str(e).lower():
            return torch.load(path, map_location=device, weights_only=False)
        raise


class Recognizer:
    """Loads a trained checkpoint. probs() returns CALIBRATED probabilities (temperature applied)."""

    def __init__(self, path, device="cpu"):
        ck = _load_checkpoint(path, device)
        self.classes, self.T, self.meta = ck["classes"], float(ck["temperature"]), ck.get("meta", {})
        self.model = ISLBiLSTM(len(self.classes), FEATURE_DIM).to(device)
        self.model.load_state_dict(ck["state"]); self.model.eval(); self.device = device

    @torch.no_grad()
    def probs(self, seq48):
        x = torch.tensor(np.asarray(seq48, np.float32)[None], device=self.device)
        return F.softmax(self.model(x) / self.T, 1)[0].cpu().numpy()

