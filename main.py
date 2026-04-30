import asyncio
import os
import textwrap
from dotenv import load_dotenv

from ai_agent.ai_agent_test import ai_agent_test

if __name__ == "__main__":
    system_ai_bot_prompt = textwrap.dedent("""
                    Ты — ИИ помощник Кемеровского Государственного Университета (КемГУ, сайт kemsu.ru).
                    ВАЖНЫЕ ПРАВИЛА:
                    1. ОТВЕЧАЙ ТОЛЬКО на основе контекста из базы знаний, который будет предоставлен ниже
                    2. ЕСЛИ В КОНТЕКСТЕ НЕТ информации для ответа — скажи: "Я не могу найти эту информацию в доступных источниках"
                    3. НЕ ПРИДУМЫВАЙ факты и не используй свои общие знания
                    4. ЦИТИРУЙ источники, если это уместно (например: "Согласно документу...")
                    5. Будь вежливым и позитивным.
                    6. БУДЬ КРАТКИМ и по существу
                """).strip()

    model_names = ["hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest", "gpt-oss:20b", "qwen3.5:9b", "phi4:14b"]

    load_dotenv()
    llm_url = os.getenv("LLM_URL")


    asyncio.run(ai_agent_test(system_ai_bot_prompt, model_names, llm_url))