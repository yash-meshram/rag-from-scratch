import sys
from src.retrieval.DataRetriever import DataRetriever
from src.generation.llm import GroqLLM
import io
import contextlib


def run_chat(data_retriever, llm):
    print("RAG Chatbot")
    print("-" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()

            if query.lower() in ["exit", "quit", "q", "e"]:
                break

            if not query:
                continue

            # generate response
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                retrieved_docs = data_retriever.retrieve(query=query)
                response = llm.generate_response(query=query, retrieved_docs=retrieved_docs)

            print(f"\nBot: {response['content']}\n(Source - {response['source']})")
            print("-" * 50)

        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    f = io.StringIO()
    db_collection_name = "pdf_documents"
    db_persist_directory = "data/VectorStore"
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        data_retriever = DataRetriever(
            collection_name=db_collection_name, persist_directory=db_persist_directory
        )
        llm = GroqLLM()
    run_chat(data_retriever=data_retriever, llm=llm)
