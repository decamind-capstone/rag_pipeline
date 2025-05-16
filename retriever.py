## 벡터 DB 검색기 
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_retriever(collection_name, db_path, model_name):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    return Chroma(
        collection_name=collection_name,
        persist_directory=db_path,
        embedding_function=embedding
    ).as_retriever()