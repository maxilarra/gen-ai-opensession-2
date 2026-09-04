from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from langgraph.graph.message import add_messages
import requests
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

@tool
def peliculas(word: str) -> str:
    """"""
    return ""

@tool
def libros(word: str) -> str:
    """"""
    return ""

@tool
def recetas(word: str) -> str:
    """"""
    return "" 

@tool
def clima(word: str) -> str:
    """"""
    return ""

def paises(word: str) -> str:
    """"""
    return ""


SYSTEM_PROMPT = """
Eres un asistente que SOLO puede responder utilizando
los resultados obtenidos de las herramientas disponibles.

No utilices tu conocimiento interno para responder.
Si ninguna herramienta puede proporcionar la información
necesaria, indica que no tienes una herramienta disponible
para obtener esa información.
No inventes ni supongas información.
"""

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano", streaming=True)

tools = [peliculas, libros, recetas, clima, paises]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Call the LLM with the current messages."""
    messages = state["messages"]
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT),*messages
    ]
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)
graph = graph_builder.compile()


def run(user_input: str):
    """Print the answer token by token, as the model writes it."""
    print()
    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Agent with tools - Console input")
    print("=" * 80)
    print("\nAvailable tools:")
    print("- peliculas")
    print("- libros")
    print("- recetas")
    print("- clima")
    print("- paises")
    print("\nType 'exit' to quit\n")

    while True:
        user_query = input("Ingrese Consulta: ").strip()
        if user_query.lower() == "exit":
            print("Chau!")
            break
        if not user_query:
            print("Por favor, introduzca una consulta válida..\n")
            continue

        run(user_query)