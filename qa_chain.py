## LLM + Retriever + RetrievalQA 체인 구성
from langchain.chains import RetrievalQA
from .prompt import technical_prompt_template

def build_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,       ## 질의에 대한 응답 + 청크 정보 함께 반환
        chain_type="stuff",                 ## 검색된 문서들을, 전부 하나의 문자열로 합쳐서 LLM에 프롬프트 식으로 넘긴다(나중에 다른 방식으로 바꾸기 ㄱㄴㄴ)
        chain_type_kwargs={"prompt": technical_prompt_template}
    )


