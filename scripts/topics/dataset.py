"""Grouped multilabel split and local training-only augmentation."""
from collections import Counter, defaultdict
import argparse
from pathlib import Path
import random

from app.topics.data import corpus_report, digest, exclusive_run, read_jsonl, redact, text_key, write_json, write_jsonl
from app.topics.taxonomy import TOPIC_NAMES, TAXONOMY_HASH, validate_labels


def split_groups(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Greedy iterative multilabel balancing of whole lineage groups.

    Ratios are targets, not an exact promise when groups cannot be divided.
    Exact text duplicates also share a component even across source groups.
    """
    if not rows:
        raise ValueError("empty corpus")
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[root(b)] = root(a)

    seen_group, seen_text, seen_near = {}, {}, {}
    cleaned = []
    labels_by_text = {}
    for i, row in enumerate(rows):
        if not row.get("group_id") or not row.get("id") or not row.get("source"):
            raise ValueError("every sample needs id, source and a known group_id")
        labels = validate_labels(row.get("labels"))
        text = redact(row["text"])
        key = text_key(text)
        if not key:
            raise ValueError("blank text")
        if key in labels_by_text and labels_by_text[key] != labels:
            raise ValueError("conflicting labels for the same cleaned text; review before splitting")
        labels_by_text[key] = labels
        for mapping, value in ((seen_group, row["group_id"]), (seen_text, key)):
            if value in mapping:
                union(i, mapping[value])
            else:
                mapping[value] = i
        if row.get('near_duplicate_group_id'):
            near=row['near_duplicate_group_id']
            if near in seen_near: union(i,seen_near[near])
            else: seen_near[near]=i
        cleaned.append({**row, "text": text, "labels": labels})
    if len({row["id"] for row in cleaned}) != len(cleaned):
        raise ValueError("duplicate sample ids")
    components = defaultdict(list)
    for i, row in enumerate(cleaned):
        components[root(i)].append(row)
    groups = list(components.values())
    for group in groups:
        component = "group-" + digest(sorted(row["id"] for row in group))[:20]
        for row in group:
            row["split_group_id"] = component
    frequency = Counter(label for group in groups for label in {x for row in group for x in row["labels"]})
    if any(frequency[label] < 10 for label in TOPIC_NAMES):
        raise ValueError("each topic needs at least 10 independent groups before 80/10/10 splitting")
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: min(frequency[label] for row in group for label in row["labels"]))
    ratios = {"train": .8, "val": .1, "test": .1}
    result = {split: [] for split in ratios}
    counts = {split: Counter() for split in ratios}
    total = Counter(label for row in cleaned for label in row["labels"])
    for group in groups:
        labels = Counter(label for row in group for label in row["labels"])

        def deficit(split):
            label_deficit = sum((total[label]*ratios[split] - counts[split][label])/total[label] for label in labels)/len(labels)
            return label_deficit + .2*(len(cleaned)*ratios[split]-len(result[split]))/len(cleaned)

        split = max(ratios, key=deficit)
        result[split].extend(group)
        counts[split].update(labels)
    if any(not counts[split][label] for split in ratios for label in TOPIC_NAMES):
        raise ValueError("whole-group balancing could not cover all topics in all splits; add independent data")
    return result


def augment_train(rows: list[dict], forbidden: set[str]) -> list[dict]:
    """Bounded local wording changes; no new facts, negation or labels."""
    output = list(rows)
    seen = forbidden | {text_key(row["text"]) for row in rows}
    for row in rows:
        text = row["text"]
        candidates = []
        for before, after in (("如何", "怎么"), ("什么时候", "啥时候"), ("可以吗", "行不行")):
            if before in text:
                candidates.append(text.replace(before, after))
        candidates.extend(("请问一下，"+text, "我想咨询一下："+text))
        for candidate in candidates:
            key = text_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            output.append({**row, "id": row["id"]+"-aug", "parent_id": row["id"], "text": candidate,
                "augmentation": "local_wording_v1"})
            break
    return output


def build(input_path: Path, output: Path, *, seed: int = 42):
    rows = read_jsonl(input_path)
    for row in rows:
        if row.get("source") == "legacy_unknown":
            raise ValueError("unknown legacy parentage cannot certify grouped holdouts; resolve provenance first")
        if row.get("label_status") not in {"human_reviewed", "assisted_review_user_approved", "reviewed_sample_batch"}:
            raise ValueError("review gate pending: record reviewer and sample-batch approval before training")
        if not row.get("reviewer"):
            raise ValueError("missing reviewer provenance")
    splits = split_groups(rows, seed)
    keys = {split: {text_key(row["text"]) for row in values} for split, values in splits.items()}
    train = augment_train(splits["train"], keys["val"] | keys["test"])
    report = {"input_sha256": digest(rows), "taxonomy_hash": TAXONOMY_HASH, "seed": seed,
        "target_ratios": [80, 10, 10], "before_augmentation": {k: corpus_report(v) for k,v in splits.items()},
        "after_augmentation": {"train": corpus_report(train), "val": corpus_report(splits["val"]), "test": corpus_report(splits["test"])},
        "augmentation": "one local wording variant per training parent only", "overlaps": {},
        "near_grouping": {"covered_samples":sum(bool(row.get('near_duplicate_group_id')) for row in rows),
            "note":"conservative lexical near-groups included in split components, not exhaustive semantic deduplication"}}
    final = {**splits, "train": train}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        text_overlap = len({text_key(row["text"]) for row in final[a]} & {text_key(row["text"]) for row in final[b]})
        group_overlap = len({row["split_group_id"] for row in final[a]} & {row["split_group_id"] for row in final[b]})
        report["overlaps"][f"{a}-{b}"] = {"texts": text_overlap, "groups": group_overlap}
        if text_overlap or group_overlap:
            raise ValueError("cross-split leakage")
    with exclusive_run(output.with_suffix(".lock")):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("datasets are frozen; use a new output directory")
        for split, values in final.items():
            write_jsonl(output/f"{split}.jsonl", values)
        write_json(output/"manifest.json", report)
    print({"sizes": {k: len(v) for k,v in final.items()}, "manifest": str(output/"manifest.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
