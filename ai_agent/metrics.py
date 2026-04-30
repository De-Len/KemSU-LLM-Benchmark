import re
from collections import Counter
import math
import torch


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred, ref):
    return int(pred.strip() == ref.strip())


def normalized_exact_match(pred, ref):
    return int(normalize_text(pred) == normalize_text(ref))


def token_f1(pred, ref):
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def jaccard_similarity(pred, ref):
    set1 = set(normalize_text(pred).split())
    set2 = set(normalize_text(ref).split())

    if not set1 and not set2:
        return 1.0

    return len(set1 & set2) / len(set1 | set2)


def extract_numbers(text):
    return re.findall(r"\d+", text)


def contains_all_numbers(pred, ref):
    ref_nums = set(extract_numbers(ref))
    pred_nums = set(extract_numbers(pred))

    if not ref_nums:
        return 1.0

    return len(ref_nums & pred_nums) / len(ref_nums)


def keyword_coverage(pred, ref):
    ref_words = set(normalize_text(ref).split())
    pred_words = set(normalize_text(pred).split())

    if not ref_words:
        return 1.0

    return len(ref_words & pred_words) / len(ref_words)


def length_ratio(pred, ref):
    if len(ref) == 0:
        return 0
    return len(pred) / len(ref)


def distinct_n(text, n=1):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return len(ngrams) / len(tokens)


def get_semantic_entropy(text):
    """Вычисляет семантическую энтропию на основе уникальности токенов"""
    tokens = text.split()
    counts = Counter(tokens)
    total = len(tokens)

    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )
    return entropy


def get_entropy_from_logprobs(logprobs):
    """Вычисляет энтропию из логарифмов вероятностей"""
    logprobs_tensor = torch.tensor(logprobs)
    probs = torch.exp(logprobs_tensor)
    entropy = -(probs * logprobs_tensor).sum()
    return entropy.item()


def get_perplexity(logprobs):
    """Вычисляет perplexity из логарифмов вероятностей"""
    return math.exp(-sum(logprobs) / len(logprobs))


def compute_logprobs_metrics(text, tokens):
    """
    Вычисляет все метрики на основе логарифмов вероятностей

    Args:
        text: сгенерированный текст
        tokens: список токенов с logprob

    Returns:
        dict: с метриками perplexity, semantic_entropy, entropy_from_logprobs
    """
    logprobs = [token["logprob"] for token in tokens]

    return {
        "perplexity": get_perplexity(logprobs),
        "semantic_entropy": get_semantic_entropy(text),
        "entropy_from_logprobs": get_entropy_from_logprobs(logprobs)
    }



def compute_all_metrics(generated_text, reference_text, tokens, scorer):
    """
    Вычисляет ВСЕ метрики для сравнения сгенерированного ответа с эталонным

    Args:
        generated_text: текст, сгенерированный моделью
        reference_text: эталонный ответ
        tokens: токены с logprob от модели
        scorer: инициализированный BERTScorer

    Returns:
        dict: полный словарь со всеми метриками
    """
    # Logprobs метрики
    logprobs_metrics = compute_logprobs_metrics(generated_text, tokens)

    # BERTScore метрики
    P, R, F1 = scorer.score([generated_text], [reference_text])

    # Текстовые метрики
    em = exact_match(generated_text, reference_text)
    norm_em = normalized_exact_match(generated_text, reference_text)
    f1_token = token_f1(generated_text, reference_text)
    jaccard = jaccard_similarity(generated_text, reference_text)

    # Числовые и ключевые метрики
    num_score = contains_all_numbers(generated_text, reference_text)
    keyword_score = keyword_coverage(generated_text, reference_text)
    len_ratio = length_ratio(generated_text, reference_text)

    # Лексическое разнообразие
    d1 = distinct_n(generated_text, 1)
    d2 = distinct_n(generated_text, 2)

    return {
        # Logprobs метрики
        "perplexity": logprobs_metrics["perplexity"],
        "semantic_entropy": logprobs_metrics["semantic_entropy"],
        "entropy_from_logprobs": logprobs_metrics["entropy_from_logprobs"],

        # BERTScore метрики
        "P": P.item(),
        "R": R.item(),
        "F1": F1.item(),

        # Текстовые метрики
        "exact_match": em,
        "normalized_exact_match": norm_em,
        "token_f1": f1_token,
        "jaccard": jaccard,

        # Содержательные метрики
        "number_match": num_score,
        "keyword_coverage": keyword_score,
        "length_ratio": len_ratio,

        # Лексические метрики
        "distinct_1": d1,
        "distinct_2": d2,

        # Дополнительные метаданные
        "text_length": len(generated_text),
        "token_count": len(tokens)
    }