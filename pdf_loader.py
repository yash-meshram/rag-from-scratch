# RAG pipeline - Data Ingestion to vector db pipeline
# --------------------------------------------------------------------------------------------------------

import sys
from ProcessDocuments import ProcessDocumentsManager
from TextSplitter import TextSplitterManager
from Embedding import EmbeddingManager

# Configure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding="utf-8")


# read all the pdf files inside the given directory (Method 2)
# --------------------------------------------------------------------------------------------------------
process_documents = ProcessDocumentsManager()
# processing all pdf in the directory
all_pdf_documents = process_documents.process_all_documents(
    documents_directory = "data/pdf_files",
    file_type = "pdf"
)
# print(all_pdf_documents[0])
# print(len(all_pdf_documents))


# Text Splitting - get teh pdf data into chunk (Method 2)
# --------------------------------------------------------------------------------------------------------
text_splitter = TextSplitterManager()
chunks = text_splitter.split_documents(all_pdf_documents)
# print(chunks[0])
# print(len(chunks))


# Embedding - convert text to vector (Method 2)
# --------------------------------------------------------------------------------------------------------
embedding_manager = EmbeddingManager()
texts = [chunk.page_content for chunk in chunks]
embeddings = embedding_manager.generate_embedding(texts)
# print(embeddings[0])
# print(len(embeddings))
