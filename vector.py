from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document
import os 
import pandas as pd 

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Check if the location exist
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

#if no prepare the data by converting it into documents
if add_documents:
    documents = []
    ids = [] 

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

# define vector store
vector_store = Chroma(
    collection_name = "restaurant_reviews",
    persist_directory = db_location,
    embedding_function = embeddings
)

# add the data into the vector store if it not already exist
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# it will grab the relevant document
retriever = vector_store.as_retriever(
    search_kwargs = {"k": 5}
)