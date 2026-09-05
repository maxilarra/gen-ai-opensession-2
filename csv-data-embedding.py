from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore 
from langchain_core.documents import Document
import re
from dotenv import load_dotenv
load_dotenv()

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

# region Creacion de documento con listado

csv_path = "imdb-top-1000.csv"
df = pd.read_csv(csv_path)

documents = []

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

    documents.append(document)

print(f"Created {len(documents)} documents.\n")

# endregion

# region Cargado del índice y prueba

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embedding_model,
    index_name=index_name
)

# endregion