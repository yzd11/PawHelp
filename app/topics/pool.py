"""Database reads/updates for offline topic classification, not chat routing."""
from collections import Counter
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert
from app.database.models import LowConfidenceQuestion, ReviewQueue, TopicClassification
from app.topics.taxonomy import TOPIC_NAMES, validate_labels
from app.topics.data import redact


class TopicPool:
    def __init__(self,sessions):
        self.sessions=sessions

    async def questions(self, *, pending: bool=False, limit: int | None=None):
        statement=select(LowConfidenceQuestion.id,LowConfidenceQuestion.raw_question,LowConfidenceQuestion.source,
            LowConfidenceQuestion.created_at,LowConfidenceQuestion.matched_review_id,
            ReviewQueue.normalized_question,ReviewQueue.occurrence_count,ReviewQueue.review_status,
            TopicClassification.labels,TopicClassification.classified_at).outerjoin(
                ReviewQueue,LowConfidenceQuestion.matched_review_id==ReviewQueue.id).outerjoin(
                TopicClassification,LowConfidenceQuestion.id==TopicClassification.question_id).order_by(LowConfidenceQuestion.id.desc())
        if pending:
            # Current classifier consumes semantically complete flywheel text.
            # Unmerged raw pronouns are not guessed as a subject.
            statement=statement.where(TopicClassification.id.is_(None),ReviewQueue.id.is_not(None))
        if limit is not None: statement=statement.limit(limit)
        async with self.sessions() as session:
            return [dict(row) for row in (await session.execute(statement)).mappings().all()]

    async def save(self,question_id: int,labels: list[str], *, reclassify: bool=False):
        labels=validate_labels(labels)
        statement=insert(TopicClassification).values(question_id=question_id,labels=labels,
            classified_at=datetime.now(timezone.utc).replace(tzinfo=None))
        if reclassify:
            statement=statement.on_duplicate_key_update(labels=statement.inserted.labels,classified_at=statement.inserted.classified_at)
        else:
            # Concurrent runners remain idempotent without refreshing timestamps.
            statement=statement.on_duplicate_key_update(question_id=statement.inserted.question_id)
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(statement)

    async def distribution(self):
        rows=await self.questions()
        classified=[row for row in rows if row["labels"] is not None]
        counts=Counter()
        unique={label:set() for label in TOPIC_NAMES}
        samples={label:[] for label in TOPIC_NAMES}
        seen={label:set() for label in TOPIC_NAMES}
        for row in classified:
            labels=validate_labels(row["labels"])
            for label in labels:
                counts[label]+=1
                unique[label].add(row["matched_review_id"] if row["matched_review_id"] else f"pool-{row['id']}")
                text=redact(row["normalized_question"] or row["raw_question"])
                if text not in seen[label] and len(samples[label])<3:
                    seen[label].add(text)
                    samples[label].append(text)
        timestamps=[row['classified_at'] for row in classified]
        return {"total":len(classified),"unclassified":len(rows)-len(classified),
            "unique_gaps_total":len({row["matched_review_id"] if row["matched_review_id"] else f"pool-{row['id']}" for row in classified}),
            "pending_merged":sum(row['labels'] is None and row['matched_review_id'] is not None for row in rows),
            "latest":max(timestamps).isoformat()+"Z" if timestamps else None,
            "hit_topics":len(counts),"topics":[{"label":label,"count":counts[label],"unique_gaps":len(unique[label]),"samples":samples[label]} for label in TOPIC_NAMES],
            "count_policy":"raw pool occurrences per label; sum may exceed total; unique_gaps uses matched review IDs"}
