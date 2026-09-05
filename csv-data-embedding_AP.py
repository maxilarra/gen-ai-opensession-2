#definidas en U 5 para busqueda vectorial
from email.mime import text

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
import os
from dotenv import load_dotenv

load_dotenv()

#se utiliza para autenticarse automáticamente en la plataforma de Hugging Face
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

    print(f"Created {len(documentos)} documentos.\n")

    vectorstore = PineconeVectorStore.from_documents(
        documents=documentos,
        embedding=embedding_model,
        index_name=index_name
    )

# endregion

# region Función de búsqueda    
def hybrid_search(query, rating=None, year=None, top_k=5):
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

    print("\n🧩 HYBRID SEARCH RESULTS")
    print("Filtro aplicado:", filter_dict)
    print(f"Cantidad de matches encontrados: {len(results['matches'])}")

    for match in results["matches"]:
        print(f"- Score: {match['score']:.4f}")
        print(f"  Title: {match['metadata'].get('title')}")
        #print(f"  Text: {match['metadata'].get('text')}")
        #print(f"  Metadata: {match['metadata']}")
        print()


def run(user_input: str):
    """Print the answer token by token, as the model writes it."""
    print()
    hybrid_search(
        query=user_input,
        rating="",
        year=2016
    )
    print("\n")



if __name__ == "__main__":
    print("=" * 80)
    print("Agent with tools - Console input")
    print("=" * 80)
    print("\nAvailable tools:")
    print("- peliculas")
    #print("- libros")
    #print("- recetas")
    #print("- clima")
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