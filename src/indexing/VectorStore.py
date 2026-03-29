# Vector Store
import os
import chromadb
from pymupdf import Document
import numpy as np
import uuid

class VectorStore():
    '''Manage embeddings in vector db like ChromaDB'''
    def __init__(self, collection_name: str, persist_directory: str):
        '''
        Args:
        collection_name = Name of the ChromaDB collection
        persist_directory = Directory to persist vector store
        '''
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.initialize_store()
        
    def initialize_store(self):
        '''Initialize ChromaDB client and collection'''
        try:
            # create the presist ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)
            
            # get or create collection
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {
                    "description": "PDF doc data(embedding converted to vector) for RAG",
                    "hnsw:space": "cosine"
                }
            )
            
            print(f"\nVector Store initialize. Collection: {self.collection_name}")
            print(f"\nDocuments in collection: {self.collection.count()}")
        except Exception as e:
            print(f"\nError initializing Vector Stor: {e}")
            raise
        
    def add_documents(self, chunks: list[Document], embeddings: np.ndarray):
        '''Add documenst(i.e chunks) and their embeddings in vector store'''
        if len(chunks) != len(embeddings):
            raise ValueError("\nNumber of documents(chunks) must mach the number of embeddings.")
        print("\nAdding documents to vector store...")
        
        # prepare data for ChromaDB
        ids = []
        documents_text = []
        embeddings_list = []
        metadatas = []
        
        # adding data to above list
        for i, (doc, embedding) in enumerate(zip(chunks, embeddings)):
            if not doc.page_content.strip():        # skip the empty page
                continue
            
            # generating unique id
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            
            # adding document
            documents_text.append(doc.page_content)
            
            # adding embedding
            embeddings_list.append(embedding.tolist())
            
            # prepare metadata and add it
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
        # add to collection
        try:
            self.collection.add(
                ids = ids,
                documents = documents_text,
                embeddings = embeddings_list,
                metadatas = metadatas
            )
            print("Successfully added documents to the vector store")
            print(f"Total documents(chunks) in collection: {self.collection.count()}")
            
            
        except Exception as e:
            print(f"Fail to add document to vector store. Error: {e}")
            raise
