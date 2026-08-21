import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()


def create_llm():
    llm = init_chat_model(
        "openai/gpt-oss-120b",
        model_provider="groq")

    return llm

