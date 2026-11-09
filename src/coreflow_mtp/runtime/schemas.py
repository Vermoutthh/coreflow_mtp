"""Semantic generation input: lengths are not token IDs or model output."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


TokenId = Annotated[StrictInt, Field(ge=0)]


class TokenizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class CacheKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: int | str
    invocation_id: int | str


class CacheLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_token_ids: list[TokenId] = Field(min_length=1)


class ToolObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    agent_id: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=128)
    duration_seconds: float = Field(ge=0)


class CacheMigrationRequest(CacheKeyRequest):
    source_url: str | None = None
    manifest: dict | None = None
    keep_cache_in_gpu: bool = True
    move: bool = False

    @model_validator(mode="after")
    def check_source(self):
        if (self.source_url is None) == (self.manifest is None):
            raise ValueError("provide exactly one source_url or exported real-tensor manifest")
        if self.move and self.source_url is None:
            raise ValueError("move requires a live source_url for acknowledged release")
        return self


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_id: str | None = Field(default=None, min_length=1, max_length=256)
    prompt: str | None = Field(default=None, min_length=1)
    prompt_token_ids: list[TokenId] | None = Field(default=None, min_length=1)
    max_tokens: int = Field(default=128, ge=1, strict=True)
    min_tokens: int = Field(default=0, ge=0, strict=True)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=-1, ge=-1, strict=True)
    seed: int | None = None
    stop: list[str] | None = None
    ignore_eos: bool = False
    agent_id: str | None = None
    workflow_id: str | None = None
    query_id: int | str | None = None
    invocation_id: int | str | None = None
    repeat: int = Field(default=0, ge=0, strict=True)
    keep_cache_in_gpu: bool = False
    free_cache: bool = True
    cache_policy: str = "manual"
    tool_name: str | None = Field(default=None, min_length=1, max_length=128)
    expected_tool_latency_seconds: float | None = Field(default=None, ge=0)
    input_semantics: str = "real_tokens"

    @model_validator(mode="before")
    @classmethod
    def accept_token_id_alias(cls, value: Any) -> Any:
        if isinstance(value, dict) and "input_token_ids" in value:
            if "prompt_token_ids" in value:
                raise ValueError("provide only one of input_token_ids and prompt_token_ids")
            value = dict(value)
            value["prompt_token_ids"] = value.pop("input_token_ids")
        return value

    @model_validator(mode="after")
    def check_prompt(self) -> GenerateRequest:
        if (self.prompt is None) == (self.prompt_token_ids is None):
            raise ValueError("provide exactly one real prompt or prompt_token_ids")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
        if self.top_k == 0:
            raise ValueError("top_k must be -1 or positive")
        if self.cache_policy not in {"manual", "predictive"}:
            raise ValueError("cache_policy must be manual or predictive")
        if self.input_semantics not in {"real_tokens", "synthetic_lengths_ar"}:
            raise ValueError("unknown input semantics")
        if self.cache_policy == "predictive" and self.keep_cache_in_gpu:
            raise ValueError("predictive cache policy decides GPU residency; do not force keep_cache_in_gpu")
        if (self.keep_cache_in_gpu or not self.free_cache) and (self.query_id is None or self.invocation_id is None):
            raise ValueError("explicit cache ownership requires query_id and invocation_id")
        if self.keep_cache_in_gpu and self.free_cache:
            raise ValueError("keep_cache_in_gpu requires free_cache=false")
        if self.stop is not None and any(not item for item in self.stop):
            raise ValueError("stop strings must not be empty")
        return self
