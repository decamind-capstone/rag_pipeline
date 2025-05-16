import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextGenerationPipeline
from langchain_huggingface import HuggingFacePipeline       ## 랭체인이 HuggingFace 파이프라인을 wrapping할 수 있도록 ㄱㄱ
from .config import LLM_MODEL_NAME

def load_llm():
    ## 4비트 양자화 위한 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

    ## 토크나이저 설정
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME, trust_remote_code=True, use_fast=False
    )       # sentencepiece 기반이므로, fast_tokeinzer를 비활성화한다
    
    ## 본격적으로 LLM 불러오기
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, 
        quantization_config = bnb_config,
        device_map='auto', 
        torch_dtype = torch.float16,
        trust_remote_code=True,
        max_memory = {0: "9GB", "cpu": "30GB"}
    )

    ## 허깅페이스 내부 TextGenerationPipeline 생성성
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # 응답 최대 길이 제한한
        temperature=0.2,    # 창의성 조절(낮을 수록 안정적)
        do_sample=True,
        top_p=0.95
    )

    pipe.task = 'text-generation'       ## 허깅페이스 파이프라인에서 task를 감지 못하는 경우에 사용...

    return HuggingFacePipeline(pipeline=pipe)
