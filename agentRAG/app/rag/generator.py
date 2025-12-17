from transformers import pipeline
from .config import GPT2_MODEL, MAX_NEW_TOKENS

generator = pipeline(
    "text-generation",
    model=GPT2_MODEL
)

def generate_answer(question, context):
    prompt = f"""
Tu es un assistant qui répond uniquement à partir du contexte fourni.

Contexte:
{context}

Question:
{question}

Réponse:
"""

    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        do_sample=True
    )

    return output[0]["generated_text"].replace(prompt, "").strip()
