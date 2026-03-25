# Text Splitting - get teh pdf data into chunk
from typing import Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextSplitterManager():
    def split_documents(
        self,
        documents: Union[Document, list[Document], list[list[Document]]], 
        chunk_size = 1000, 
        chunk_overlap = 200
        ) -> list[Document]:
        '''split documents into smaller chunk for better rag performance'''
        if isinstance(documents, Document):
            flat_docs = [documents]
        elif isinstance(documents, list) and all(isinstance(doc, list) for doc in documents):
            flat_docs = [d for doc in documents for d in doc]
        else:
            flat_docs = documents
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )
        splits_docs = text_splitter.split_documents(flat_docs)
        print(f"Splitted {len(flat_docs)} pages into {len(splits_docs)} chunks.")
        return splits_docs