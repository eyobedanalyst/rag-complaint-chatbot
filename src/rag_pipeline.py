# src/rag_pipeline.py

from typing import List, Tuple

# -------------------------
# Prompt Template
# -------------------------
RAG_PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Answer the question using ONLY the complaint excerpts provided below.
If the information is not present in the context, say you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""


# -------------------------
# Helper Functions
# -------------------------
def truncate_context(context: str, max_tokens: int = 400) -> str:
    tokens = context.split()
    return " ".join(tokens[:max_tokens])


def build_context(retrieved_chunks: List[dict]) -> str:
    return "\n\n".join(
        [f"- {chunk['text']}" for chunk in retrieved_chunks]
    )


# -------------------------
# Retriever
# -------------------------
def retrieve_relevant_chunks(
    query: str,
    collection,
    embedding_model,
    k: int = 5
) -> List[dict]:
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )

    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text": doc,
            "metadata": meta
        })

    return retrieved


# -------------------------
# Generator (RAG Core)
# -------------------------
def generate_rag_answer(
    question: str,
    collection,
    embedding_model,
    llm,
    k: int = 5
) -> Tuple[str, List[dict]]:
    retrieved = retrieve_relevant_chunks(
        query=question,
        collection=collection,
        embedding_model=embedding_model,
        k=k
    )

    if not retrieved:
        return "I do not have enough information to answer this question.", []

    context = build_context(retrieved)
    context = truncate_context(context)

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = llm(prompt)[0]["generated_text"].strip()

    return response, retrieved
