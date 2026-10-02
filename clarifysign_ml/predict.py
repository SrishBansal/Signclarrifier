import numpy as np, torch, torch.nn.functional as F
from .model import ISLBiLSTM
from .features import FEATURE_DIM


class Recognizer:
    """Loads a trained checkpoint. probs() returns CALIBRATED probabilities (temperature applied)."""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        self.classes, self.T, self.meta = ck["classes"], float(ck["temperature"]), ck.get("meta", {})
        self.model = ISLBiLSTM(len(self.classes), FEATURE_DIM).to(device)
        self.model.load_state_dict(ck["state"]); self.model.eval(); self.device = device

    @torch.no_grad()
    def probs(self, seq48):
        x = torch.tensor(np.asarray(seq48, np.float32)[None], device=self.device)
        return F.softmax(self.model(x) / self.T, 1)[0].cpu().numpy()
