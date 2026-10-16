"""Build an auditable sample pack; never auto-claim human approval."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from datetime import datetime, timezone
from app.topics.data import corpus_report, digest, read_jsonl, write_json, write_jsonl
from app.topics.taxonomy import TOPIC_NAMES, TAXONOMY_HASH, validate_labels


def prepare(real: Path, synthetic: Path, output: Path):
    rows=read_jsonl(real)+read_jsonl(synthetic)
    if len({row['id'] for row in rows})!=len(rows): raise ValueError("duplicate sample IDs")
    manifest=json.loads(synthetic.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    generated=[row for row in rows if row['source']=='synthetic']
    if len(generated)!=manifest['per_topic']*len(TOPIC_NAMES):
        raise ValueError("generation is not complete; cannot freeze a partial corpus for review")
    if any(validate_labels(row['labels'])!=row['labels'] for row in rows):
        raise ValueError("labels must be valid and canonical before review")
    rng=random.Random(42)
    selected={row['id']:row for row in rows if row['source']=='real_pool'}
    for label in TOPIC_NAMES:
        candidates=[row for row in generated if label in row['labels']]
        selected.update({row['id']:row for row in rng.sample(candidates,min(5,len(candidates)))})
    pack={"approved":False,"reviewer":None,"taxonomy_hash":TAXONOMY_HASH,
        "input_hash":digest(rows),"real_path":str(real),"synthetic_path":str(synthetic),
        "policy":"all real records + at least five generated positives per topic; no automatic human approval",
        "corpus_report":corpus_report(rows),
        "items":[{"id":row['id'],"source":row['source'],"text":row['text'],"proposed_labels":row['labels'],
            "accepted_labels":None,"note":""} for row in selected.values()]}
    if output.exists(): raise FileExistsError("review packs are versioned; never overwrite annotations")
    write_json(output,pack)
    lines=["# 分类器预标抽审", "",f"候选 {len(rows)} 条；真实 {len(rows)-len(generated)}，模拟 {len(generated)}；抽审 {len(selected)} 条。",
        "其他整句兜底；只标明确诉求；修归保修维修，普通退换归退换货；商品回收归商品信息。",
        "这是预标候选，尚未获人工批准。错误请按样本ID指出；JSON中的accepted_labels填写审定标签。", "",
        "| ID | 来源 | 问题 | 预标标签 |", "|---|---|---|---|"]
    for row in pack['items']:
        lines.append(f"|{row['id']}|{row['source']}|{row['text'].replace('|','／')}|{'、'.join(row['proposed_labels'])}|")
    output.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print({"review_samples":len(selected),"candidate_samples":len(rows),"review_pack":str(output)})


def confirm(pack_path: Path, corrections_path: Path, output: Path, *, reviewer: str, approval: str):
    """Snapshot explicit user approval of an assisted review, without changing originals."""
    if not reviewer.strip() or not approval.strip():
        raise ValueError("explicit reviewer and user approval quotation required")
    pack=json.loads(pack_path.read_text(encoding='utf-8'))
    if pack['approved'] or pack['taxonomy_hash']!=TAXONOMY_HASH:
        raise ValueError("expected pending review of current taxonomy")
    rows=read_jsonl(Path(pack['real_path']))+read_jsonl(Path(pack['synthetic_path']))
    if digest(rows)!=pack['input_hash']: raise ValueError("candidate corpus changed after sample review")
    by_id={row['id']:row for row in rows}
    selected={item['id']:item for item in pack['items']}
    if len(selected)!=len(pack['items']): raise ValueError("duplicate review IDs")
    for item in selected.values():
        original=by_id[item['id']]
        if item['text']!=original['text'] or item['proposed_labels']!=original['labels']:
            raise ValueError("review sample no longer matches original")
        item['accepted_labels']=validate_labels(item['proposed_labels'])
    corrections=json.loads(corrections_path.read_text(encoding='utf-8'))
    seen=set()
    for correction in corrections:
        sample_id=correction['id']
        if sample_id not in by_id or sample_id in seen:
            raise ValueError("unknown or duplicate correction ID")
        seen.add(sample_id)
        original=by_id[sample_id]
        item=selected.setdefault(sample_id, {"id":sample_id,"source":original['source'],
            "text":original['text'],"proposed_labels":original['labels']})
        if 'add_labels' in correction and 'labels' not in correction:
            labels=set(original['labels'])|set(validate_labels(correction['add_labels']))
            item['accepted_labels']=validate_labels([label for label in TOPIC_NAMES if label in labels])
        elif 'labels' in correction and 'add_labels' not in correction:
            item['accepted_labels']=validate_labels(correction['labels'])
        else:
            raise ValueError("correction must specify either labels or add_labels")
        item['note']=correction['note']
    pack.update(approved=True,reviewer=reviewer,review_method='codex_assisted_user_approved',
        approval_quote=approval,approved_at=datetime.now(timezone.utc).isoformat(),
        original_pack_hash=digest(json.loads(pack_path.read_text(encoding='utf-8'))),
        corrections_hash=digest(corrections),items=list(selected.values()))
    if output.exists(): raise FileExistsError("approved review packs are immutable")
    write_json(output,pack)
    print({"approved_samples":len(selected),"corrections":len(seen),"pack":str(output)})


def apply(pack_path: Path, output: Path):
    pack=json.loads(pack_path.read_text(encoding='utf-8'))
    if pack['approved'] is not True or not isinstance(pack['reviewer'],str) or not pack['reviewer'].strip():
        raise ValueError("human review pending; do not train")
    if pack['taxonomy_hash']!=TAXONOMY_HASH: raise ValueError("taxonomy changed after review")
    rows=read_jsonl(Path(pack['real_path']))+read_jsonl(Path(pack['synthetic_path']))
    if digest(rows)!=pack['input_hash']: raise ValueError("candidate corpus changed after sample review")
    decisions={row['id']:validate_labels(row['accepted_labels']) for row in pack['items']}
    if len(decisions)!=len(pack['items']): raise ValueError("duplicate review IDs")
    if not set(decisions)<= {row['id'] for row in rows}: raise ValueError("unknown review ID")
    for row in rows:
        if row['id'] in decisions:
            row['labels']=decisions[row['id']]
            row['label_status']='assisted_review_user_approved' if pack.get('review_method')=='codex_assisted_user_approved' else 'human_reviewed'
        else:
            row['label_status']='reviewed_sample_batch'
        row.update(reviewer=pack['reviewer'],review_pack_hash=digest(pack),
            review_method=pack.get('review_method','human_review'),approval_quote=pack.get('approval_quote'))
    if output.exists(): raise FileExistsError("approved corpora are frozen; use a new output")
    write_jsonl(output,rows)
    print({"approved":len(rows),"reviewed_samples":len(decisions),"output":str(output)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','confirm','apply'])
    parser.add_argument('--real',type=Path,default=Path('data/topic-classifier/pool-prelabels.jsonl'))
    parser.add_argument('--synthetic',type=Path,default=Path('data/topic-classifier/synthetic-v1.jsonl'))
    parser.add_argument('--pack',type=Path,default=Path('data/topic-classifier/sample-review.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--corrections',type=Path)
    parser.add_argument('--reviewer')
    parser.add_argument('--approval')
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.real,args.synthetic,args.pack)
    elif args.action=='confirm':
        if not all((args.output,args.corrections,args.reviewer,args.approval)):
            parser.error('--output, --corrections, --reviewer and --approval required')
        confirm(args.pack,args.corrections,args.output,reviewer=args.reviewer,approval=args.approval)
    else:
        if not args.output: parser.error('--output required')
        apply(args.pack,args.output)


if __name__=='__main__':main()
