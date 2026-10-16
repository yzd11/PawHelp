"""Offline corpus contracts, local redaction and a persistent paid-call budget."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

from app.topics.taxonomy import TAXONOMY_HASH, TOPIC_NAMES, validate_labels


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def exclusive_run(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink()


def redact(text: str) -> str:
    """Run BEFORE any external model request; do not repair language/negations."""
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "<邮箱>", text)
    text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "<证件>", text)
    text = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "<电话>", text)
    text = re.sub(r"(?:微信|微信号|QQ|qq)\s*[:：]?\s*[A-Za-z0-9_-]{5,}", "<联系方式>", text)
    text = re.sub(r"((?:订单(?:号)?|单号|运单号|快递单号)\s*[:：#]?\s*)[A-Za-z0-9-]{4,}", r"\1<编号>", text)
    text = re.sub(r"(?<!\d)\d{4,20}\s*(?:号)?订单", "<编号>订单", text)
    text = re.sub(r"(?<!\d)\d{12,}(?!\d)", "<长编号>", text)
    text = re.sub(r"(?:收件人|收货人|联系人|姓名)\s*[:：]\s*[^\s，,。；;]{1,30}", "<姓名>", text)
    text = re.sub(r"(?:收货地址|收件地址|地址)\s*[:：]\s*[^\n。；;]{1,150}", "<地址>", text)
    return re.sub(r"\s+", " ", text)


def text_key(text: str) -> str:
    text = redact(text).lower()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


class PaidBudget:
    """No automatic retries; failed/uncertain requests also consume the budget."""
    def __init__(self, path: Path, limit: int = 100):
        if not 1 <= limit <= 100:
            raise ValueError("first-round authorized budget is at most 100 requests")
        self.path, self.limit = path, limit

    def reserve(self, purpose: str, payload: object) -> str:
        with exclusive_run(self.path.with_suffix(".lock")):
            state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"limit": self.limit, "calls": []}
            if state["limit"] != self.limit:
                raise ValueError("cannot reset/raise an existing authorized budget")
            if len(state["calls"]) >= self.limit:
                raise RuntimeError("paid request budget exhausted; ask user before continuing")
            call_id = str(len(state["calls"]) + 1)
            state["calls"].append({"id": call_id, "purpose": purpose, "input_hash": digest(payload),
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "reserved"})
            write_json(self.path, state)
            return call_id

    def finish(self, call_id: str, *, status: str, usage: dict | None) -> None:
        with exclusive_run(self.path.with_suffix(".lock")):
            state = json.loads(self.path.read_text(encoding="utf-8"))
            row = next(row for row in state["calls"] if row["id"] == call_id)
            row.update(status=status, usage=usage, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(self.path, state)


def corpus_report(rows: list[dict]) -> dict:
    counts = {label: 0 for label in TOPIC_NAMES}
    invalid = []
    for i, row in enumerate(rows):
        try:
            for label in validate_labels(row.get("labels")):
                counts[label] += 1
        except ValueError as exc:
            invalid.append({"index": i, "reason": str(exc)})
    return {"size": len(rows), "label_counts": counts, "invalid": invalid,
        "taxonomy_hash": TAXONOMY_HASH, "sha256": digest(rows)}
