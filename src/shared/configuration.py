"""Configuration shared across indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration for the AI Research Assistant."""

    embedding_model: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={
            "description": "Local HuggingFace embedding model."
        },
    )

    retriever_provider: Literal["chroma"] = field(
        default="chroma",
        metadata={
            "description": "Local ChromaDB vector store."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"k": 8},
        metadata={
            "description": "Arguments passed to the ChromaDB retriever."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T],
        config: Optional[RunnableConfig] = None,
    ) -> T:
        """Create configuration from a LangGraph RunnableConfig."""

        config = ensure_config(config)
        configurable = config.get("configurable") or {}

        _fields = {
            f.name
            for f in fields(cls)
            if f.init
        }

        return cls(
            **{
                key: value
                for key, value in configurable.items()
                if key in _fields
            }
        )


T = TypeVar("T", bound=BaseConfiguration)