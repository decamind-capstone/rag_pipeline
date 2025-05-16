# ## Utils.py 기반으로 벡터 DB 구축하는 모듈
# from utils import build_chroma_from_chunks

# # for group, json_path, name in targets:
# #     build_chroma_from_chunks(group, json_path, name)


# if __name__ == "__main__":
#     print("벡터 DB 생성 시작!!")

#     targets = [
#         ('MSA', './data/json_chunks/MSA/8071_hierarchical_structure.json', 'sff_8071'),
#         # ('MSA', './data/json_chunks/MSA/8071_hierarchical_structure.json', 'sff_8071')
#     ]

#     for group, json_path, name in targets:
#         build_chroma_from_chunks(group, json_path, name)


from utils import build_chroma_from_chunks
from pathlib import Path
import os

if __name__ == "__main__":
    print("📂 current working dir:", os.getcwd())
    print("📄 __file__ path:", os.path.abspath(__file__))
    targets = [
        (
            'MSA',
            './data/json_chunks/MSA/8071_hierarchical_structure.json',
            'sff_8071'
        ),
        # 추가 문서가 있다면 여기에 계속 append 가능
    ]

    for group, json_path, collection_name in targets:
        print(f"\n📌 Now building collection: {collection_name}")
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"❌ JSON 파일 없음: {json_path}")
            continue

        try:
            build_chroma_from_chunks(group, str(json_path), collection_name)
        except Exception as e:
            print(f"❗️에러 발생: {e}")