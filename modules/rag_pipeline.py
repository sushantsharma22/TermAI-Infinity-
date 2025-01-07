from modules.retrieval import LocalDocRetriever
from modules.text_generation import generate_text

class RAGPipeline:
    """
    Minimal retrieval-augmented generation pipeline:
    1) Retrieve relevant text from local files.
    2) Combine it with user's question.
    3) Generate final answer using local LLM.
    """

    def __init__(self):
        self.retriever = LocalDocRetriever()

    def ask_question(self, question: str) -> str:
        context = self.retriever.get_relevant_text(question, k=3)
        if not context.strip():
            prompt = f"Answer the question:\nQuestion: {question}\nAnswer:"
        else:
            prompt = (
                f"You have the following context:\n\n"
                f"{context}\n\n"
                f"Use this context to answer the question:\n{question}\nAnswer:"
            )
        answer = generate_text(prompt, max_length=200)
        return answer
