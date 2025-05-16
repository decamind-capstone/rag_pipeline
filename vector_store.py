## RAG 시스템에서 벡터 DB를 로드 및 저장, 문서 추가
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

def load_vector_db(persist_dir: str, embedding_model_name: str, collection_name: str):
    """특정 경로에서 ChromaDB를 불러온다!!(기존 벡터DB 재사용)"""
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if not os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        raise FileNotFoundError(f"Chroma DB not found at {persist_dir}")
    
    return Chroma(
        persist_directory=persist_dir,
        embedding_function = embedding,
        collection_name=collection_name
    )

def save_vector_db(chroma_db):
    """현재 ChromaDB 객체 상태를 디스크 내에 저장 ㄱㄱㄱ"""
    chroma_db.persist()
    chroma_db = None


def add_documents(chroma_db, texts, metadatas):
    """벡터 DB 내에 새로운 문서 청크를 추가한다"""
    chroma_db.add_texts(texts=texts, metadatas=metadatas)
    chroma_db.persist()  # 문서 추가 후 꼭 저장