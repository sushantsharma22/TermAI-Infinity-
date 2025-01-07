import os
from modules.text_generation import generate_text
from modules.prompts import chunk_summary_template, combine_summaries_template

class Summarizer:
    """
    Chunk-based summarization for large files:
    1) Split the file into smaller chunks.
    2) Summarize each chunk individually.
    3) Combine chunk summaries into a final summary.
    """

    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size

    def summarize_file(self, filepath: str) -> str:
        """
        Summarizes a text file in chunks, then merges partial summaries.
        """
        if not os.path.exists(filepath):
            return "File not found."

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self._split_into_chunks(text, self.chunk_size)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = chunk_summary_template.format(chunk_text=chunk)
            summary = generate_text(prompt, max_length=150)
            chunk_summaries.append(summary.strip())

        # Combine partial summaries
        combined_prompt = combine_summaries_template.format(partial_summaries="\n".join(chunk_summaries))
        final_summary = generate_text(combined_prompt, max_length=200)
        return final_summary

    def _split_into_chunks(self, text: str, size: int):
        """
        Split text into chunks of approximate length 'size'.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end
        return chunks
