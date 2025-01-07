import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Default local model
MODEL_NAME = os.environ.get("TERM_AI_TEXT_MODEL", "gpt2")

print(f"[Text Generation] Using local model: {MODEL_NAME}")

def _init_text_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

_text_pipeline = _init_text_pipeline()

def generate_text(prompt: str, max_length: int = 100) -> str:
    """
    Generate text using the local pipeline with default or user-specified decoding parameters.
    """
    output = _text_pipeline(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.9
    )
    return output[0]["generated_text"]
