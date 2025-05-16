"""LLM 프롬프트 생성 전용 유틸리티
사용자의 자연어 질문과 검색된 context를 기반으로 다양한 형태의 프롬프트를 생성한다."""

def build_prompt(query: str, context: str, mode: str = "default") -> str:
    """
    기본 프롬프트 생성기
    :param query: 사용자의 질문
    :param context: 검색된 문서 context
    :param mode: 'default' or 'kor' 등
    """
    if mode == "default":
        return f"""[CONTEXT]
{context}

[USER QUESTION]
{query}

[ANSWER]"""
    elif mode == "kor":
        return f"""[문서 내용]
{context}

[질문]
{query}

[답변 (자연어로 요약)]:"""
    else:
        raise ValueError("지원하지 않는 프롬프트 모드입니다.")

def build_cot_prompt(query: str, context: str) -> str:
    """
    Chain-of-Thought 유도 프롬프트
    """
    return f"""[문서 내용]
{context}

[질문]
{query}

[답변은 다음과 같은 방식으로 단계별로 사고해서 작성하세요]
1. 문서 내용 중 관련 근거를 식별
2. 근거를 바탕으로 논리적으로 생각 전개
3. 최종 결론을 도출

[답변]:"""

def build_multidoc_prompt(query: str, contexts: list[str]) -> str:
    """
    Top-k 문서 context 병합용 프롬프트
    """
    context_combined = "\n---\n".join(contexts)
    return f"""[다수 문서 요약]
{context_combined}

[질문]
{query}

[요약된 정보에 기반한 답변]:"""

def build_visual_hint_prompt(query: str, context: str, figures: str = "") -> str:
    """
    OCR 기반 도표 포함 프롬프트
    """
    return f"""[문서 내용]
{context}

[도표/그림 설명 (OCR 추출)]
{figures}

[질문]
{query}

[답변]:"""
