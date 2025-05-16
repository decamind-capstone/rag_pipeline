from pathlib import Path

BASE_PDF_DIR = Path('./data/pdf_docs')
BASE_VECDB_DIR = Path('./data/vector_dbs')

# PDF 그룹 정의
PDF_GROUPS = ['MSA', 'Military Standard']

# PDF 경로 정의
PDF_PATHS = {
    group: list((BASE_PDF_DIR / group).glob('*.pdf')) for group in PDF_GROUPS
}

VECTOR_DB_PATHS = {
    group: BASE_VECDB_DIR / group for group in PDF_GROUPS
}

EMBEDDING_MODELS = 'intfloat/e5-base'
LLM_MODEL_NAME = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"


# # 여기 안의 PDF 경로들?
# MSA_PDFS = list(MSA_DIR.glob('*.pdf'))

# # 파일명 -> 컬렉션명 매핑
# COLLECTION_MAP = {
#     path.name: path.stem.lower().replace(' ', '_') for path in MSA_PDFS
# }


# def get_collection_name(file_path: str | Path) -> str:
#     from config import COLLECTION_MAP
#     filename = Path(file_path).name

#     if filename not in COLLECTION_MAP:
#         raise ValueError(f"Collection name not found for {filename}")
#     return COLLECTION_MAP[filename]