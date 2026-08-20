"""Researcher graph used in the conversational retrieval system as a subgraph.

This module defines the core structure and functionality of the researcher graph,
which is responsible for generating search queries and retrieving relevant documents.
"""

from typing import TypedDict, cast

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.state import QueryState, ResearcherState
from shared import retrieval
from shared.utils import load_chat_model


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries for a research step."""

    class Response(TypedDict):
        queries: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)

    model = load_chat_model(
        configuration.query_model
    ).with_structured_output(Response)

    messages = [
        {
            "role": "system",
            "content": configuration.generate_queries_system_prompt,
        },
        {
            "role": "human",
            "content": state.question,
        },
    ]

    response = cast(Response, await model.ainvoke(messages))

    print("\n--- GENERATED SEARCH QUERIES ---")
    for i, query in enumerate(response["queries"], start=1):
        print(f"{i}. {query}")

    return {"queries": response["queries"]}


async def retrieve_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve relevant documents for a search query."""

    print("\n--- RETRIEVING FOR QUERY ---")
    print(state.query)

    with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(
            state.query,
            config,
        )
        
    print(f"Retrieved {len(response)} documents")

    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_documents", QueryState(query=query)) for query in state.queries
    ]


# Define the graph
builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_documents"],
)
builder.add_edge("retrieve_documents", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "ResearcherGraph"
