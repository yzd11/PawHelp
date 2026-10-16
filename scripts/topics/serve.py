"""Independent classifier: bind 127.0.0.1:8110; no realtime chat coupling."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints
from app.topics.model import TopicModel


Text = Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=512)]


class ClassificationRequest(BaseModel):
    texts: list[Text] = Field(min_length=1,max_length=40)


def create_app(model_dir: Path | None = None, loader=TopicModel):
    @asynccontextmanager
    async def lifespan(app):
        app.state.classifier=loader(model_dir or Path(os.getenv("TOPIC_MODEL_DIR","data/topic-classifier/model")))
        yield
        del app.state.classifier

    app=FastAPI(title="PawHelp offline topic classifier",lifespan=lifespan)

    @app.get("/healthz")
    def health():
        metadata=app.state.classifier.metadata
        return {"status":"ok","model_version":metadata["model_version"],"taxonomy_hash":metadata["taxonomy_hash"]}

    @app.post("/classify")
    def classify(request: ClassificationRequest):
        return {"items":app.state.classifier.classify(request.texts)}

    return app


app=create_app()
