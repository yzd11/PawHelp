"""Full-parameter RoBERTa training and frozen held-out evaluation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from app.topics.data import digest, exclusive_run, read_jsonl, write_json
from app.topics.taxonomy import ID2LABEL, LABEL2ID, TAXONOMY_HASH, TOPIC_NAMES
from app.topics.metrics import evaluate, evaluate_sources, labels_from_scores, scan_threshold, sigmoid

BASE_MODEL = "hfl/chinese-roberta-wwm-ext"


def stage_base(output: Path):
    """Download official weights into workspace and witness real model loss.

    This is initialization, never a fine-tuned model or a quality result.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("base staging directory already populated; inspect rather than overwrite")
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,cache_dir="data/topic-classifier/hf-cache")
    model=AutoModelForSequenceClassification.from_pretrained(BASE_MODEL,cache_dir="data/topic-classifier/hf-cache",
        num_labels=len(TOPIC_NAMES),problem_type="multi_label_classification",id2label=ID2LABEL,label2id=LABEL2ID)
    if not torch.cuda.is_available(): raise RuntimeError("authorized CUDA unavailable")
    model.to("cuda")
    tokens=tokenizer(["买大了想退"],return_tensors="pt",truncation=True,max_length=128).to("cuda")
    labels=torch.zeros((1,len(TOPIC_NAMES)),device="cuda",dtype=torch.float32)
    result=model(**tokens,labels=labels)
    result.loss.backward()
    if not torch.isfinite(result.loss): raise RuntimeError("actual BERT multilabel loss nonfinite")
    revision=getattr(model.config,"_commit_hash",None)
    model.to("cpu").save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    write_json(output/"base-source.json",{"source":BASE_MODEL,"revision":revision,"taxonomy_hash":TAXONOMY_HASH,
        "witness":"OFFICIAL_BASE_MULTILABEL_FORWARD_BACKWARD_PASS","shape":list(result.logits.shape),
        "loss":result.loss.item(),"fine_tuned":False})
    print(json.dumps({"base":BASE_MODEL,"revision":revision,"witness":"OFFICIAL_BASE_MULTILABEL_FORWARD_BACKWARD_PASS","fine_tuned":False}))


def preflight(output: Path):
    import torch
    spec=json.loads(Path("config/topic-training.json").read_text(encoding="utf-8"))
    versions={name:importlib.metadata.version(name) for name in ("torch","transformers","accelerate")}
    if versions!=spec["packages"]:
        raise RuntimeError("compute spec differs from actual versions; review environment before training")
    if not torch.cuda.is_available():
        raise RuntimeError("authorized local CUDA GPU unavailable; do not silently switch training device")
    torch.manual_seed(42)
    x = torch.randn(32,32,device="cuda",requires_grad=True)
    loss = (x@x).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA forward/backward witness failed")
    report = {"witness": "CUDA_FORWARD_BACKWARD_PASS", "device": torch.cuda.get_device_name(),
        "loss": loss.item(), "versions": versions, "spec_hash":digest(spec)[:8], "timestamp": datetime.now(timezone.utc).isoformat()}
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False))


def train(dataset: Path, output: Path, *, epochs: int, batch: int):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the authorized local training")
    if not 1 <= epochs <= 8 or not 1 <= batch <= 8:
        raise ValueError("epochs 1..8 and microbatch 1..8 only")
    manifest = json.loads((dataset/"manifest.json").read_text(encoding="utf-8"))
    if manifest["taxonomy_hash"] != TAXONOMY_HASH or any(x["texts"] or x["groups"] for x in manifest["overlaps"].values()):
        raise ValueError("dataset taxonomy/leakage gate failed")
    rows = {split: read_jsonl(dataset/f"{split}.jsonl") for split in ("train","val")}
    for split in rows:
        if digest(rows[split]) != manifest["after_augmentation"][split]["sha256"]:
            raise ValueError("frozen dataset changed after manifest generation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("never overwrite trained models; use a new output directory")
    torch.manual_seed(42)
    base=Path("data/topic-classifier/base")
    source=json.loads((base/"base-source.json").read_text(encoding="utf-8"))
    if source["source"]!=BASE_MODEL or source["taxonomy_hash"]!=TAXONOMY_HASH:
        raise ValueError("staged base does not match fixed technology/taxonomy")
    tokenizer = AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(base),local_files_only=True, num_labels=len(TOPIC_NAMES),
        id2label=ID2LABEL, label2id=LABEL2ID, problem_type="multi_label_classification")
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("full parameter training requires every parameter trainable")

    class Encoded(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            row = self.items[i]
            encoded = tokenizer(row["text"], truncation=True, max_length=128, padding="max_length")
            return {**encoded, "labels": [float(label in row["labels"]) for label in TOPIC_NAMES]}

    def collate(items):
        return {key: torch.tensor([row[key] for row in items], dtype=torch.float32 if key=="labels" else torch.long) for key in items[0]}

    def metrics(prediction):
        expected = [[label for label, bit in zip(TOPIC_NAMES,row) if bit] for row in prediction.label_ids.tolist()]
        predicted = [labels_from_scores([sigmoid(x) for x in row], .5) for row in prediction.predictions.tolist()]
        report = evaluate(expected, predicted)
        return {"micro_f1": report["micro"]["f1"], "macro_f1": report["macro"]["f1"]}

    with exclusive_run(output.with_suffix(".lock")):
        args = TrainingArguments(output_dir=str(output/"checkpoints"), num_train_epochs=epochs,
            per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
            gradient_accumulation_steps=8, gradient_checkpointing=True, fp16=True,
            learning_rate=2e-5, weight_decay=.01, warmup_steps=20, max_grad_norm=1.0,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="micro_f1", greater_is_better=True,
            logging_steps=20, dataloader_num_workers=0, report_to="none", seed=42)
        trainer = Trainer(model=model, args=args, train_dataset=Encoded(rows["train"]),
            eval_dataset=Encoded(rows["val"]), processing_class=tokenizer, data_collator=collate,
            compute_metrics=metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
        trainer.train()
        result = trainer.predict(Encoded(rows["val"]))
        scores = [[sigmoid(x) for x in row] for row in result.predictions.tolist()]
        scan = scan_threshold([row["labels"] for row in rows["val"]], scores)
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        metadata = {"base_model": BASE_MODEL, "base_revision": source["revision"],
            "model_version": digest({"dataset": manifest, "best": trainer.state.best_model_checkpoint,
                "run": datetime.now(timezone.utc).isoformat()})[:20], "taxonomy_hash": TAXONOMY_HASH,
            "labels": list(TOPIC_NAMES), "threshold": scan["chosen"]["threshold"], "max_length":128,
            "selection": "best validation micro-F1 checkpoint; validation-only threshold scan",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "dataset": str(dataset.resolve()), "manifest_hash":digest(manifest),
            "dataset_hashes": {split: item["sha256"] for split,item in manifest["after_augmentation"].items()},
            "policy": "specific labels over threshold, exclusive other fallback; no argmax invented topic",
            "versions": {name: importlib.metadata.version(name) for name in ("torch","transformers","accelerate")}}
        write_json(output/"threshold.json", metadata)
        write_json(output/"threshold-scan.json", {**scan,"validation_scores": scores,"validation_labels":[row["labels"] for row in rows["val"]]})
        write_json(output/"training-history.json", {"history": trainer.state.log_history,
            "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric})
        print(json.dumps({"model_version":metadata["model_version"], "threshold": metadata["threshold"]},ensure_ascii=False))


def evaluate_model(model_dir: Path, dataset: Path, output: Path):
    from app.topics.model import TopicModel
    model = TopicModel(model_dir)
    metadata = model.metadata
    manifest = json.loads((dataset/"manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(dataset/"test.jsonl")
    if digest(manifest) != metadata["manifest_hash"] or digest(rows) != metadata["dataset_hashes"]["test"]:
        raise ValueError("held-out test data no longer matches trained model lineage")
    predicted = []
    for start in range(0,len(rows),16):
        predicted.extend(model.classify([row["text"] for row in rows[start:start+16]]))
    report = evaluate([row["labels"] for row in rows], [row["labels"] for row in predicted])
    errors = []
    for row, result in zip(rows,predicted):
        expected, actual = set(row["labels"]), set(result["labels"])
        if expected != actual:
            errors.append({"id":row["id"],"text":row["text"],"expected":row["labels"],"predicted":result["labels"],
                "missed":sorted(expected-actual),"extra":sorted(actual-expected),"scores":result["scores"],
                "review_status":"pending","source":row["source"]})
    report.update(model_version=metadata["model_version"],threshold=metadata["threshold"],taxonomy_hash=TAXONOMY_HASH,
        test_sha256=digest(rows),source_counts=dict(__import__('collections').Counter(row['source'] for row in rows)),
        by_source=evaluate_sources([row['labels'] for row in rows],[row['labels'] for row in predicted],
                                  [row['source'] for row in rows]),
        interpretation="fixed held-out dataset scores; synthetic scores are not production-real accuracy")
    with exclusive_run(output.with_suffix(".lock")):
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/"eval-report.json",report)
        write_json(output/"error-samples.json",errors)
        lines = ["# 多标签主题分类测试报告",f"模型：{metadata['model_version']}；测试：{len(rows)}条；阈值：{metadata['threshold']}",
            "", "| 类目 | P | R | F1 | support | TN/FP/FN/TP |", "|---|---:|---:|---:|---:|---|"]
        for item in report["per_class"]:
            matrix=item['matrix']
            lines.append(f"|{item['label']}|{item['precision']:.4f}|{item['recall']:.4f}|{item['f1']:.4f}|{item['support']}|{matrix[0][0]}/{matrix[0][1]}/{matrix[1][0]}/{matrix[1][1]}|")
        lines.extend(["",f"micro-F1={report['micro']['f1']:.4f}，macro-F1={report['macro']['f1']:.4f}",
            "", "| 来源 | 条数 | micro-F1 | macro-F1 |", "|---|---:|---:|---:|"])
        for source,item in report['by_source'].items():
            lines.append(f"|{source}|{item['size']}|{item['micro']['f1']:.4f}|{item['macro']['f1']:.4f}|")
        lines.append("小样本成绩不代表生产准确率；缺失类目的macro-F1记0，不据极小真实子集宣称真实泛化。错例待复核。")
        (output/"eval-report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"micro":report["micro"],"macro":report["macro"],"errors":len(errors)},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["preflight","stage-base","train","evaluate"])
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--batch",type=int,default=2)
    args=parser.parse_args()
    if args.action=="preflight":
        preflight(args.output)
    elif args.action=="stage-base":
        stage_base(args.output)
    elif args.action=="train":
        if not args.dataset: parser.error("--dataset required")
        train(args.dataset,args.output,epochs=args.epochs,batch=args.batch)
    else:
        if not args.dataset or not args.model: parser.error("--dataset and --model required")
        evaluate_model(args.model,args.dataset,args.output)


if __name__ == "__main__":
    main()
