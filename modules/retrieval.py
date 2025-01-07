import os
import glob
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class LocalDocRetriever:
    """
    Loads text files from 'data/' folder, embeds them, and provides a similarity_search for RAG.
    """

    def __init__(self, data_dir="data", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.vectorstore = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        text_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        if not text_files:
            print(f"[RAG] No text files found in '{self.data_dir}'. RAG context is empty.")
            return

        # Load documents
        docs = []
        for file_path in text_files:
            loader = TextLoader(file_path, encoding="utf-8")
            file_docs = loader.load_and_split()
            docs.extend(file_docs)

        # Create embeddings & vector store
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = Chroma.from_documents(docs, embeddings, collection_name="term_ai_infinity_docs")
        print(f"[RAG] Indexed {len(docs)} chunk(s) from local text files.")

    def get_relevant_text(self, query: str, k: int = 3) -> str:
        """
        Return top-k relevant chunks as a combined string.
        """
        if not self.vectorstore:
            return ""
        results = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in results])
