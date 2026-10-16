"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    AnyHttpUrl,
    Field,
    PositiveInt,
    SecretStr,
    StringConstraints,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings for the customer service API."""

    chat_base_url: AnyHttpUrl = Field(
        validation_alias=AliasChoices("CHAT_BASE_URL", "OPENAI_BASE_URL")
    )
    openai_model: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    openai_api_key: SecretStr
    database_url: SecretStr
    shop_name: str = "本店"
    model_timeout_seconds: PositiveInt = 60
    context_token_budget: PositiveInt = 4096
    model_max_output_tokens: PositiveInt = Field(default=1024, validation_alias=AliasChoices("MAX_OUTPUT_TOKENS", "MODEL_MAX_OUTPUT_TOKENS"))
    model_context_window: int = Field(default=0, ge=0)
    max_user_input_tokens: PositiveInt = 2000
    max_agent_steps: PositiveInt = 3
    tool_result_max_tokens: PositiveInt = 3000
    rerank_top_k: PositiveInt = 5
    evidence_chunk_max_tokens: PositiveInt = 600
    context_safety_tokens: PositiveInt = 1000
    context_target_turns: PositiveInt = 20
    context_chars_per_token: float = Field(default=1.2, ge=0.5, le=4)
    context_layer2_reply_chars: PositiveInt = 60
    context_layer2_tool_tokens: PositiveInt = 200
    context_summary_max_tokens: PositiveInt = 1000
    context_summary_max_chars: PositiveInt = 200
    context_summary_timeout_seconds: PositiveInt = 60
    tool_timeout_seconds: PositiveInt = 60
    tool_max_attempts: PositiveInt = 2
    mcp_enabled: bool = True
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
    mcp_discovery_timeout_seconds: float = Field(default=3, gt=0, le=30)
    mcp_demo_secret: SecretStr = SecretStr("local-demo-only-change-me")
    tool_policy_path: str = "config/tool-policy.json"
    tool_plugins_path: str = "tool_plugins"
    structured_output_method: Literal["function_calling", "json_mode", "json_schema"] = "json_mode"
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: PositiveInt = 16
    embedding_max_length: PositiveInt = 1024
    vectorize_batch_size: PositiveInt = 16
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    milvus_hybrid_collection: str = "knowledge_hybrid"
    knowledge_similarity_threshold: float = Field(default=0.45, ge=-1, le=1)
    knowledge_rerank_threshold: float = Field(default=0.1, ge=0, le=1)
    dense_recall_limit: PositiveInt = 50
    bm25_recall_limit: PositiveInt = 50
    rerank_limit: PositiveInt = 10
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: PositiveInt = 8
    reranker_query_max_length: PositiveInt = 256
    reranker_passage_max_length: PositiveInt = 1024
    conversation_mining_batch_size: PositiveInt = 20
    observability_enabled: bool = True
    langfuse_enabled: bool = False
    langfuse_base_url: str = "http://127.0.0.1:3000"
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    admin_token: SecretStr = SecretStr("")
    flywheel_interval_seconds: int = Field(default=0, ge=0)
    flywheel_batch_size: int = Field(default=5, ge=1, le=50)
    flywheel_candidate_limit: int = Field(default=100, ge=1, le=300)
    evaluation_interval_seconds: int = Field(default=0, ge=0)
    evaluation_sample_per_bucket: int = Field(default=2, ge=1, le=60)
    evaluation_max_model_calls: int = Field(default=100, ge=1, le=5000)
    evidence_calibration_path: str = "config/evidence-calibration.json"
    observation_snapshot_limit: int = Field(default=5, ge=1, le=20)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Load and cache production settings from the environment."""

    return Settings()
