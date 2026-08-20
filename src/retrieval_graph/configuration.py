"""Configuration for the AI Research Assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from retrieval_graph import prompts
from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """Configuration for the AI Research Assistant."""

    # LLM models

    query_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "llm"}},
    ] = field(
        # default="google_genai/gemini-3.6-flash",
        default="google_genai/gemini-3.1-flash-lite",
        metadata={
            "description": "Gemini model used for query analysis and research planning."
        },
    )

    response_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "llm"}},
    ] = field(
        # default="google_genai/gemini-3.6-flash",
        default="google_genai/gemini-3.1-flash-lite",
        metadata={
            "description": "Gemini model used to generate the final answer."
        },
    )

    # Prompts

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "Prompt used to classify the user's question."
        },
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={
            "description": "Prompt used when additional information is required."
        },
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "Prompt used for general questions."
        },
    )

    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "Prompt used to create the research plan."
        },
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "Prompt used to generate multiple search queries."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={
            "description": "Prompt used to generate the final grounded answer."
        },
    )