from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu

def compute_bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu([reference_tokens], hypothesis_tokens)

def compute_bert_score(reference: str, hypothesis: str, lang='en') -> float:
    P, R, F1 = bert_score([hypothesis], [reference], lang=lang)
    return F1[0].item()

def hybrid_score(reference: str, hypothesis: str, alpha=0.5) -> float:
    bleu = compute_bleu(reference, hypothesis)
    bert = compute_bert_score(reference, hypothesis)
    return alpha * bleu + (1 - alpha) * bert