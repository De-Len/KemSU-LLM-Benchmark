import json

import httpx

class LLMClient:
    def __init__(
        self,
        url,
        model,
        api_key="",
        timeout=120,
        max_tokens=16000
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)
        self.max_tokens = max_tokens

    def _payload(self, system, user, max_tokens, logprobs, stream=False):
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            # "max_tokens": max_tokens,
            "logprobs": logprobs,
            "stream": stream,
        }

    async def chat(self, system, user, logprobs):
        r = await self.client.post(
            self.url,
            json=self._payload(system, user, self.max_tokens, logprobs, False),
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def stream(self, system, user, logprobs):
        async with self.client.stream(
            "POST",
            self.url,
            json=self._payload(system, user, self.max_tokens, logprobs, True),
            headers={"Authorization": f"Bearer {self.api_key}"}
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta

    async def generate_stream(self, system, user, logprobs=False, max_response: int = 10000):
        """
        Стриминговая генерация с ограничением по длине ответа.
        Поддержка logprobs (потоковая, в каждом чанке).
        """
        data = self._payload(system, user, self.max_tokens, logprobs, True)

        full_message = ""
        all_logprobs = []

        try:
            async with self.client.stream(
                    "POST",
                    self.url,
                    json=data,
            ) as response:

                if response.status_code != 200:
                    text = await response.aread()
                    error_msg = f"Ошибка {response.status_code}: {text.decode()}"
                    return {"text": error_msg, "logprobs": []}

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    chunk = json.loads(line)

                    # ===== контент =====
                    if "message" in chunk:
                        content = chunk["message"].get("content")
                        if content:
                            full_message += content

                    # ===== logprobs (в каждом чанке) =====
                    if logprobs and "logprobs" in chunk:
                        for token_info in chunk["logprobs"]:
                            all_logprobs.append({
                                "token": token_info.get("token"),
                                "logprob": token_info.get("logprob"),
                                "top_logprobs": []  # можно оставить пустым
                            })

                    # ===== завершение =====
                    if chunk.get("done"):
                        break

                    # ограничение длины
                    if len(full_message) >= max_response:
                        print(f"Достигнут лимит max_response={max_response}")
                        break

            print(f"Размер message: {len(full_message)}")
            print(f"Количество logprobs: {len(all_logprobs)}")

            return {
                "text": full_message,
                "logprobs": all_logprobs
            }

        except httpx.TimeoutException:
            return {"text": "Ошибка: Timeout при обращении к LLM", "logprobs": []}
        except Exception as e:
            return {"text": f"Ошибка при обращении к LLM: {str(e)}", "logprobs": []}

    async def close(self):
        await self.client.aclose()