from pymupdf import Document
from src.indexing.ProcessDocuments import ProcessDocumentsManager
from src.indexing.TextSplitter import TextSplitterManager
from src.indexing.Embedding import EmbeddingManager
from src.indexing.VectorStore import VectorStore
import numpy as np
from typing import Union

class DataIngestion():
    '''
    RAG pipeline - Data Ingestion to vector db pipeline
    '''
    
    def __init__(self):
        self.all_documents = None
        self.chunks = None
        self.embeddings = None
        
    def process_documents(self, documents_directory: str, file_type: str = "pdf"):
        '''
        read all the documents files inside the given directory
        '''
        process_documents = ProcessDocumentsManager()
        # processing all documents in the directory
        self.all_documents = process_documents.process_all_documents(
            documents_directory,
            file_type
        )
        return self.all_documents

    def split_documents(
        self, 
        all_documents: Union[Document, list[Document], list[list[Document]]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
        ) -> list[Document]:
        '''
        Text Splitting - get the pdf data into chunk
        '''
        text_splitter = TextSplitterManager()
        self.chunks = text_splitter.split_documents(all_documents, chunk_size=1000, chunk_overlap=200)
        return self.chunks

    def generate_embedding(
        self, 
        chunks: list[Document],
        model_name: str = "multi-qa-MiniLM-L6-cos-v1"
        ) -> np.ndarray:
        '''Embedding - convert text to vector
        '''
        embedding_manager = EmbeddingManager(model_name=model_name)
        texts = [chunk.page_content for chunk in chunks]
        self.embeddings = embedding_manager.generate_embedding(texts)
        return self.embeddings

    def add_documents_to_db(
        self, 
        chunks: list[Document], 
        embeddings: np.ndarray,
        collection_name: str = "pdf_documents", 
        persist_directory: str = "data/VectorStore"
        ):
            '''
            Vector Store - adding data in vector db
            '''
            vector_store = VectorStore(
                collection_name = collection_name, 
                persist_directory = persist_directory
            )
            vector_store.add_documents(
                chunks=chunks, 
                embeddings=embeddings
            )
