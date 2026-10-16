"""Bounded standalone pool batch, persistent inference provenance, no LLM calls."""
import argparse
import asyncio
from datetime import datetime, timezone
import math
from pathlib import Path
import uuid
import httpx

from app.config import get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.topics.data import digest, exclusive_run, redact, write_json
from app.topics.pool import TopicPool
from app.topics.taxonomy import TAXONOMY_HASH, TOPIC_NAMES, validate_labels


async def classify(output: Path, *, force: bool, min_batch: int, limit: int):
    engine=create_async_engine_from_settings(get_settings())
    try:
        pool=TopicPool(create_session_factory(engine))
        with exclusive_run(output/"batch.lock"):
            rows=await pool.questions(pending=True,limit=limit)
            run={"run_id":uuid.uuid4().hex,"started_at":datetime.now(timezone.utc).isoformat(),
                "status":"started","pending":len(rows),"classified":[],"failed":[],"taxonomy_hash":TAXONOMY_HASH}
            path=output/f"classify-{run['run_id']}.json"
            write_json(path,run)
            if len(rows)<min_batch and not force:
                run["status"]="waiting_for_batch"
                write_json(path,run)
                print(f"waiting: eligible={len(rows)}, min_batch={min_batch}")
                return
            if not rows:
                run["status"]="no_pending"
                write_json(path,run)
                print("no pending merged questions; unchanged results")
                return
            async with httpx.AsyncClient(timeout=60,trust_env=False) as client:
                health=(await client.get("http://127.0.0.1:8110/healthz"))
                health.raise_for_status()
                meta=health.json()
                if meta.get("taxonomy_hash")!=TAXONOMY_HASH: raise ValueError("classifier taxonomy mismatch")
                run["model_version"]=meta["model_version"]
                for start in range(0,len(rows),20):
                    batch=rows[start:start+20]
                    texts=[redact(row["normalized_question"]) for row in batch]
                    response=await client.post("http://127.0.0.1:8110/classify",json={"texts":texts})
                    response.raise_for_status()
                    items=response.json()["items"]
                    if len(items)!=len(batch): raise ValueError("classifier response count mismatch")
                    for row,text,item in zip(batch,texts,items):
                        labels=validate_labels(item["labels"])
                        scores=item["scores"]
                        if item.get("model_version")!=meta["model_version"] or item.get("taxonomy_hash")!=TAXONOMY_HASH:
                            raise ValueError("classifier model changed during batch")
                        if set(scores)!=set(TOPIC_NAMES) or any(not isinstance(s,(int,float)) or not math.isfinite(s) or not 0<=s<=1 for s in scores.values()):
                            raise ValueError("invalid classifier scores")
                        # Record an uncertain write BEFORE SQL so a failed log
                        # never silently loses the model/input provenance.
                        record={"question_id":row['id'],"input_hash":digest(text),"labels":labels,"scores":scores,"status":"write_pending"}
                        run["classified"].append(record)
                        write_json(path,run)
                        await pool.save(row['id'],labels)
                        record['status']='saved'
                        write_json(path,run)
                    print(f"classified={len(run['classified'])}/{len(rows)}",flush=True)
            run.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat())
            write_json(path,run)
    finally:
        await engine.dispose()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--min-batch",type=int,default=10)
    parser.add_argument("--limit",type=int,default=100)
    parser.add_argument("--output",type=Path,default=Path("data/topic-classifier/reports/batches"))
    args=parser.parse_args()
    if not 1<=args.min_batch<=args.limit<=500: parser.error("1 <= min-batch <= limit <= 500 required")
    asyncio.run(classify(args.output,force=args.force,min_batch=args.min_batch,limit=args.limit))


if __name__=="__main__": main()
