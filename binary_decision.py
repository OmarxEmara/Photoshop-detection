import json, math, os
from pathlib import Path

_DEFAULT_CFG = {
    "higher_is_fraud": True,
    "threshold": 0.50,  # fallback if no config file or env var
}

class DecisionEngine:
    def __init__(self, threshold: float, higher_is_fraud: bool = True):
        self.threshold = float(threshold)
        self.higher_is_fraud = bool(higher_is_fraud)

    @classmethod
    def from_json(cls, path: str = "config/decision_config.json"):
        # env override wins
        env_thr = os.getenv("FRAUD_THRESHOLD")
        env_dir = os.getenv("FRAUD_SCORE_DIRECTION", "").lower().strip()  # "higher" or "lower"
        cfg = dict(_DEFAULT_CFG)

        p = Path(path)
        if p.exists():
            with open(p, "r") as f:
                file_cfg = json.load(f)
                cfg.update(file_cfg or {})

        if env_thr:
            try:
                cfg["threshold"] = float(env_thr)
            except ValueError:
                pass

        if env_dir in ("higher", "higher_is_fraud", "high"):
            cfg["higher_is_fraud"] = True
        elif env_dir in ("lower", "lower_is_fraud", "low"):
            cfg["higher_is_fraud"] = False

        return cls(cfg["threshold"], cfg["higher_is_fraud"])

    def decide(self, fraud_score: float) -> str:
        # SAFE default if score is NaN/missing -> FAIL conservatively
        try:
            s = float(fraud_score)
        except Exception:
            return "fail"

        if math.isnan(s):
            return "fail"

        if self.higher_is_fraud:
            return "fail" if s > self.threshold else "pass"
        else:
            return "fail" if s < self.threshold else "pass"
