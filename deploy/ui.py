"""
Gradio chat interface para el Edge Chatbot (Jais 7B).
Conecta directo a llama.cpp server (OpenAI-compatible API).

Uso:
    python deploy/ui.py
"""

import os

import gradio as gr
import httpx

LLAMA_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
SYSTEM_PROMPT = os.getenv(
    "CHATBOT_SYSTEM_PROMPT",
    "أنت مساعد افتراضي ودود متخصص في الشؤون العائلية، تتحدث باللهجة الإماراتية. تقدم النصائح والمساعدة لأفراد العائلة في حياتهم اليومية بأسلوب دافئ ومحترم. أنت دائماً مساعد وليس فرداً من العائلة. لا تتقمص دور أم أو أب أو جد أو أي قريب. استخدم التعبيرات الإماراتية الشائعة وحافظ على أسلوب مهذب ومهني.",
)


def chat_fn(message: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        resp = httpx.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={"messages": messages, "temperature": 0.7, "max_tokens": 512},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        return f"خطأ في الاتصال بالخادم: {e}"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="🇦🇪 مساعد العائلة الإماراتي",
    description="شات بوت عائلي باللهجة الإماراتية — Jais 7B fine-tuned, Q5_K_M, CPU-only",
    examples=[
        "هلا يمه شخبارك؟",
        "شو نطبخ حق الغدا اليوم؟",
        "ولدي ما يبي يروح المدرسة شو أسوي؟",
        "نبي نسوي يمعة عائلية هالويكند",
        "علميني أسوي لقيمات",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
