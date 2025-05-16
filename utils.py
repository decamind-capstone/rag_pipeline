# ## PDF 파일 -> 청킹 & JSON 변환 -> 벡터 DB 저장장
# import fitz
# import re

# def extract_and_clean_text(pdf_filepath):
#     """지정된 PDF 문서에서 텍스트 추출 및 정제"""
#     try:
#         docs = fitz.open(pdf_filepath)
#         extracted_text_all_pages = ""
#         start_page_no = 13  # 13페이지부터 추출

#         for page_no in range(start_page_no - 1, docs.page_count):
#             page = docs.load_page(page_no)
#             extracted_text_all_pages += page.get_text("text")
#             extracted_text_all_pages += f"\n--- End of Page {page_no + 1} ---\n\n"

#         cleaned_text = extracted_text_all_pages

#         # 정규 표현식 패턴
#         patterns = [
#             r"^\s*Published\s+SFF-8071 Rev 1\.7\s*$",
#             r"^\s*SFP\+ 1X 0\.8mm Card Edge Connector\s+Page\s+\d+\s*$",
#             r"^\s*SFP\+ 1X 0\.8mm Card Edge Connector\s*$",
#             r"^\s*--- End of Page \d+ ---\s*$",
#             r'\n\s*\n'  # 여러 빈 줄
#         ]

#         for pattern in patterns:
#             cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.MULTILINE)

#         cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text) # 3개 이상의 빈 줄을 2개로 축소
#         cleaned_text = cleaned_text.strip()

#         return cleaned_text

#     except Exception as e:
#         print(f"'{pdf_filepath}' 파일 처리 중 오류 발생: {e}")
#         return ""

# if __name__ == '__main__':
#     # 테스트용 PDF 파일 경로
#     pdf_file = 'C:/Users/OesCnuChatbot/Desktop/AI학습자료 for OES/decamind_000/capstone_design/fastapi_app/app/rag_pipeline/data/pdf_docs/MSA/SFF-8472 Rev 12.2.pdf'  # 실제 파일 경로로 변경해 주세요!
    
#     # 텍스트 추출 및 정제
#     extracted_text = extract_and_clean_text(pdf_file)

#     if extracted_text:
#         # 추출된 텍스트 출력 (또는 파일에 저장)
#         # print("--- 추출된 텍스트 ---\n", extracted_text)  # 너무 길면 주석 처리
        
#         # 파일에 저장하는 경우
#         output_file = 'extracted_text.txt'
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(extracted_text)
#         print(f"추출된 텍스트가 '{output_file}' 파일에 저장되었습니다.")
#     else:
#         print(f"'{pdf_file}' 에서 텍스트 추출 실패.")

## 기존 .json 파일 청킹화하기
import json
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODELS, VECTOR_DB_PATHS

def load_chunks_from_json(json_path: str) -> list[dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def build_chroma_from_chunks(group: str, json_path: str, collection_name: str):
    chunks = load_chunks_from_json(json_path)

    texts = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            'section_code': chunk['section_code'],
            'depth': chunk['depth'],
            'hierarchy': json.dumps(chunk['hierarchy']),
            'section_path': json.dumps(chunk['section_path'])
        } for chunk in chunks
    ]

    # 로그 확인하기
    print(f"총 청크 수: {len(texts)}")
    print(f"첫 청크 내용: {texts[0][:50]}")
    print(f"첫 메타데이터: {metadatas[0]}")

    vecdb_path = VECTOR_DB_PATHS[group]
    vecdb_path.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS)

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=str(vecdb_path),
        collection_name=collection_name,
        metadatas=metadatas
    )

    db.persist()
    print(f"[✅] Saved collection '{collection_name}' to {vecdb_path}")