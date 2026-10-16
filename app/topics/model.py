"""Model loaded only by the independent offline classifier process."""
import json
from pathlib import Path
from threading import Lock

from app.topics.data import redact
from app.topics.metrics import labels_from_scores
from app.topics.taxonomy import TAXONOMY_HASH, TOPIC_NAMES


class TopicModel:
    def __init__(self, path: Path, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.metadata=json.loads((path/"threshold.json").read_text(encoding="utf-8"))
        if self.metadata["taxonomy_hash"] != TAXONOMY_HASH or self.metadata["labels"] != list(TOPIC_NAMES):
            raise ValueError("frozen model taxonomy mismatch")
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(str(path),local_files_only=True).to(device).eval()
        if [self.model.config.id2label[i] for i in range(len(TOPIC_NAMES))] != list(TOPIC_NAMES):
            raise ValueError("model output label mapping mismatch")
        self.device, self.torch, self.lock = device, torch, Lock()

    def classify(self, texts: list[str]) -> list[dict]:
        if not texts or len(texts)>40 or any(not isinstance(t,str) or not t.strip() or len(t)>512 for t in texts):
            raise ValueError("1..40 nonblank texts, at most 512 characters each")
        with self.lock, self.torch.inference_mode():
            tokens=self.tokenizer([redact(t) for t in texts],padding=True,truncation=True,
                max_length=self.metadata["max_length"],return_tensors="pt").to(self.device)
            scores=self.model(**tokens).logits.sigmoid().cpu().tolist()
        threshold=self.metadata["threshold"]
        return [{"labels":labels_from_scores(row,threshold),"scores":dict(zip(TOPIC_NAMES,row)),
            "fallback":not any(score>=threshold for score in row[:-1]),
            "model_version":self.metadata["model_version"],"taxonomy_hash":TAXONOMY_HASH} for row in scores]
