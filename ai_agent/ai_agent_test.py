import json
import time

from bert_score import BERTScorer

from ai_agent.metrics import compute_all_metrics
from infrastructure.llm_client import LLMClient
from logger import log_metrics


def build_system_prompt(base_prompt, chunks):
    """Формирует системный промпт с контекстом из базы знаний"""
    return f"""
    {base_prompt} 
    \nКОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
    \n{chr(10).join(chunks)} 
    \nИспользуй ТОЛЬКО этот контекст для ответа. 
    \nЕсли нужной информации нет в контексте, следуй правилу 2 из системных инструкций.
    """


async def process_dialogue(llm_client, system_ai_bot_prompt, scorer, model_name, dialogue):
    """Обрабатывает один диалог: получает ответ от модели и вычисляет метрики"""

    dialogue_id = dialogue['id']
    user_question = dialogue['messages'][0]['text']
    reference_answer = dialogue['messages'][1]['text']
    chunks = dialogue['messages'][0]['chunks']

    print(f"\nДиалог ID: {dialogue_id}")
    print(f"  Вопрос пользователя: {user_question}")
    print(f"  Правильный ответ: {reference_answer[:100]}...")

    system_prompt = build_system_prompt(system_ai_bot_prompt, chunks)

    start_time = time.perf_counter()
    response = await llm_client.generate_stream(
        system_prompt,
        user_question,
        logprobs=True
    )
    elapsed = time.perf_counter() - start_time

    generated_text = response['text']
    tokens = response['logprobs']

    print(f"  Ответ модели: {generated_text[:150]}...")
    print(f"  Время ответа: {elapsed:.2f} сек")

    metrics = compute_all_metrics(
        generated_text=generated_text,
        reference_text=reference_answer,
        tokens=tokens,
        scorer=scorer
    )

    full_metrics = {
        "model": model_name,
        "id": dialogue_id,
        "question": user_question,
        "output": generated_text,
        "latency_sec": elapsed,
        **metrics
    }

    print(f"  F1: {metrics['F1']:.4f} | Perplexity: {metrics['perplexity']:.4f} | "
          f"Token F1: {metrics['token_f1']:.4f} | Keyword coverage: {metrics['keyword_coverage']:.4f}")

    return full_metrics


async def ai_agent_test(system_ai_bot_prompt: str, model_names: [str], llm_url: str):
    """Основная функция тестирования"""

    with open('test_dialog.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    scorer = BERTScorer(lang="ru", model_type="bert-base-multilingual-cased")

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"МОДЕЛЬ: {model_name}")
        print(f"{'=' * 60}")

        llm_client = LLMClient(model=model_name, url=llm_url)

        for category in data['conversations']:
            for dialogue in category['dialogues']:
                metrics = await process_dialogue(
                    llm_client=llm_client,
                    system_ai_bot_prompt=system_ai_bot_prompt,
                    scorer=scorer,
                    model_name=model_name,
                    dialogue=dialogue
                )
                log_metrics(metrics)

        await llm_client.close()
        print(f"\nЗавершено тестирование модели: {model_name}")