import json
from datetime import datetime
from pathlib import Path

METRICS_DIR = Path("metrics_logs")
METRICS_DIR.mkdir(exist_ok=True)

LOG_PATH = METRICS_DIR / "llm_metrics.jsonl"

def log_metrics(data: dict):
    data["timestamp"] = datetime.utcnow().isoformat()

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")