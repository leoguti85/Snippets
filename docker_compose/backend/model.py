from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)


def load_llm():
    model_id_llama = "alokabhishek/Llama-2-7b-chat-hf-bnb-4bit"

    tokenizer_llama = AutoTokenizer.from_pretrained(model_id_llama, use_fast=True)

    model_llama = AutoModelForCausalLM.from_pretrained(
        model_id_llama, device_map="auto"
    )

    pipe_llama = pipeline(
        model=model_llama, tokenizer=tokenizer_llama, task="text-generation"
    )

    return pipe_llama
