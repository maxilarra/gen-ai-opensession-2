from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from langgraph.graph.message import add_messages
import requests
from sympy.strategies import condition
from typing_extensions import TypedDict


from sentence_transformers import SentenceTransformer # type: ignore
from huggingface_hub import login

#definidas en U 5 para cargar indice
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore 
from langchain_core.documents import Document
import re


#importaciones comunes
# panda para trabajar con csv
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")



# @tool
# def libros(query: str) -> str:
#     """Busca información relacionada con libros."""
#     return ""


# @tool
# def recetas(query: str) -> str:
#     """Busca información relacionada con recetas de cocina."""
#     return ""


# @tool
# def paises(query: str) -> str:
#     """Busca información relacionada con países."""
#     return ""

# lo referente a la carga de índice de películas se encuentra en csv-data-embedding_AP.py   
login(token=os.getenv("HF_TOKEN"))

# region Configuración de Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "movies-for-open-session"

pc = Pinecone(api_key=PINECONE_API_KEY)


if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Configuración de Pinecone completada")

# endregion

# region Cargado del índice y prueba Unid 5 Demo 5

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = pc.Index(index_name)
index_stats = index.describe_index_stats()

print(index_stats["total_vector_count"])

if index_stats["total_vector_count"] > 0:
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embedding_model,
        index_name=index_name
    )
    print("Entro por indice existente")
else:
    csv_path = "imdb-top-1000.csv"
    df = pd.read_csv(csv_path)

    documentos = []

    for _, movie in df.iterrows():
        document = f"""
        Title: {movie['Series_Title']}
        Release Year: {movie['Released_Year']}
        Certificate: {movie['Certificate']}
        Runtime: {movie['Runtime']}
        Genre: {movie['Genre']}
        IMDb Rating: {movie['IMDB_Rating']}
        Meta Score: {movie['Meta_score']}
        Director: {movie['Director']}
        Stars: {movie['Star1']}, {movie['Star2']}, {movie['Star3']}, {movie['Star4']}
        Votes: {movie['No_of_Votes']}
        Gross Revenue: {movie['Gross']}
        Overview:
        {movie['Overview']}
        """.strip()

        doc = Document(
            page_content=document,
            metadata={
                "title": movie["Series_Title"],
                "year": str(movie["Released_Year"]),
                "genre": movie["Genre"],
                "rating": float(movie["IMDB_Rating"]) if pd.notna(movie["IMDB_Rating"]) else None,
                "director": movie["Director"],
            },
        )

        documentos.append(doc)

    #print(f"Created {len(documentos)} documentos.\n")

    vectorstore = PineconeVectorStore.from_documents(
        documents=documentos,
        embedding=embedding_model,
        index_name=index_name
    )

# endregion
# region Función de búsqueda    
def hybrid_search(query, rating=None, year=None, top_k=5) -> str:

    query_embedding = embedding_model.embed_query(query)

    # Construcción del filtro metadata
    filter_dict = {}

    if rating:
        filter_dict["rating"] = {"$eq": rating}

    if year:
        filter_dict["year"] = {"$eq": f"{year}"}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    #print("\n🧩 HYBRID SEARCH RESULTS")
    #print("Filtro aplicado:", filter_dict)
    #print(f"Cantidad de matches encontrados: {len(results['matches'])}")

    if not results["matches"]:
            return "No se encontraron películas relacionadas."

    # for match in results["matches"]:
    #     #print(f"- Score: {match['score']:.4f}")
    #     salida =print(f"  Title: {match['metadata'].get('title')}")
    #     #print(f"  Text: {match['metadata'].get('text')}")
    #     #print(f"  Metadata: {match['metadata']}")
    #     #print()

    # #salida =print(f"  Title: {re.match['metadata'].get('title')}")

    # return salida

    titulos = [match["metadata"].get("title") for match in results["matches"]]

    return ", ".join(titulos)

def busco_peliculas(user_input: str) -> str:
    """Print the answer token by token, as the model writes it."""
    #print()
    # resultado_consulta = hybrid_search(
    #     query=user_input,
    #     rating="",
    #     year=""
    # )
    resultado_consulta = hybrid_search(query=user_input)
    #print("\n")
    return resultado_consulta


@tool
def peliculas(user_query: str) -> str:
   """Busca información relacionada con películas."""
   salida_gral = busco_peliculas(user_query)   
#    return salida_gral
   return f"La pelicula recomendada es: {salida_gral}"


@tool
def clima(city: str) -> str:
    """
    Devuelve el clima actual para una ciudad usando Open-Meteo.
    """
    # First, get coordinates from Open-Meteo geocoding API
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_resp = requests.get(geo_url).json()

    if "results" not in geo_resp or len(geo_resp["results"]) == 0:
        return f"Could not find coordinates for {city}"

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]

    # Get current weather
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_resp = requests.get(weather_url).json()

    if "current_weather" not in weather_resp:
        return f"Weather data unavailable for {city}"

    weather = weather_resp["current_weather"]
    temp = weather["temperature"]
    wind = weather["windspeed"]
    condition = weather.get("weathercode", "unknown")

    return f"Current weather in {city}: {temp}°C, wind {wind} km/h, condition code {condition}"



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

# tools = [peliculas, libros, recetas, clima, paises]
tools = [peliculas, clima]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Call the LLM with the current messages."""
    messages = state["messages"]
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT),*messages]
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


# def run(user_input: str):
#     result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
#     print(f"\n{result['messages'][-1].content}\n")


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
    # print("- libros")
    # print("- recetas")
    print("- clima")
    #print("- paises")
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