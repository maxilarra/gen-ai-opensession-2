from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, List
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

products = [
    {"id": 1, "nombre": "Teclado mecánico Logitech G413", "categoria": "Teclados", "precio": 1850.0, "stock": 12},
    {"id": 2, "nombre": "Teclado Redragon Kumara K552", "categoria": "Teclados", "precio": 950.0, "stock": 0},
    {"id": 3, "nombre": "Mouse Logitech G305", "categoria": "Mouses", "precio": 780.0, "stock": 25},
    {"id": 4, "nombre": "Mouse Razer DeathAdder Essential", "categoria": "Mouses", "precio": 650.0, "stock": 8},
    {"id": 5, "nombre": "Monitor Samsung Odyssey 27\"", "categoria": "Monitores", "precio": 4500.0, "stock": 5},
    {"id": 6, "nombre": "Monitor LG UltraGear 24\"", "categoria": "Monitores", "precio": 3200.0, "stock": 0},
    {"id": 7, "nombre": "Monitor Dell S2721DGF 27\"", "categoria": "Monitores", "precio": 6100.0, "stock": 3},
    {"id": 8, "nombre": "Auriculares HyperX Cloud II", "categoria": "Audio", "precio": 1650.0, "stock": 15},
    {"id": 9, "nombre": "Parlantes Logitech Z313", "categoria": "Audio", "precio": 890.0, "stock": 0},
    {"id": 10, "nombre": "Notebook Lenovo IdeaPad 3", "categoria": "Notebooks", "precio": 14500.0, "stock": 4},
]

@tool
def calculate_total_cost(prices: List[float]) -> float:
    """
    Calculate the total cost of a purchase given a list of item prices.
    Returns the sum of all prices, rounded to 2 decimals.
    """
    return round(sum(prices), 2)


@tool
def calculate_shipping_cost(purchase_amount: float) -> float:
    """
    Calculate the shipping cost based on the total purchase amount.
    - Less than $1,000 -> $120 shipping.
    - Between $1,000 and $5,000 (inclusive) -> $60 shipping.
    - More than $5,000 -> free shipping ($0).
    """
    if purchase_amount < 1000:
        return 120
    elif purchase_amount <= 5000:
        return 60
    else:
        return 0


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")

tools = [calculate_total_cost, calculate_shipping_cost]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Call the LLM with the current messages."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
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
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    print(f"\n{result['messages'][-1].content}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Agent with Tools - Console Input")
    print("=" * 80)
    print("\nAvailable tools:")
    print("- Calculate total cost from a list of prices")
    print("- Calculate shipping cost based on purchase amount")
    print("\nType 'exit' to quit\n")

    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.\n")
            continue

        run(user_query)