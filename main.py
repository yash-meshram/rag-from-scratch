from src.indexing.DataIngestion import DataIngestion
from src.retrieval.DataRetriever import DataRetriever
from src.generation.llm import GroqLLM
import sys

# Configure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding="utf-8")

# data_ingestion = DataIngestion()
db_collection_name = "pdf_documents"
db_persist_directory = "data/VectorStore"
data_retriever = DataRetriever(
    collection_name = db_collection_name,
    persist_directory = db_persist_directory
)
llm = GroqLLM()

# # load all pdf files in teh give directory
# all_pdf_documents = data_ingestion.process_documents(
#     documents_directory="data/pdf_files",
#     file_type="pdf"
# )
# # print(all_pdf_documents[0])
# print(f"all_pdf_documents = {len(all_pdf_documents)}")
# print(f"{type(all_pdf_documents)}")


# # split the documents
# chunks = data_ingestion.split_documents(
#     all_documents=all_pdf_documents
# )
# # print(chunks[0])
# print(f"chunks = {len(chunks)}")
# print(f"{type(chunks)}")


# # embedding
# embeddings = data_ingestion.generate_embedding(
#     chunks=chunks
# )
# # print(embeddings[0])
# print(f"embeddings = {len(embeddings)}")
# print(f"{type(embeddings)}")


# # add data to vector db
# data_ingestion.add_documents_to_db(
#     chunks=chunks,
#     embeddings=embeddings,
#     collection_name=db_collection_name,
#     persist_directory=db_persist_directory
# )


query = "English Constituency Parsing"

# retriever
retrieved_docs = data_retriever.retrieve(
    query = query
)
# print(retrieved_docs)
# print(f"retrieved docs = {len(retrieved_docs)}")
# print(type(retrieved_docs))

# llm - getting response
response = llm.generate_response(
    query = query,
    retrieved_docs = retrieved_docs
)
# print(response)
print(f"\nResponse = {response["content"]}")
print(f"\nSource = {response["source"]}")
# print(type(response))

