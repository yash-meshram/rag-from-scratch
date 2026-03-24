# RAG pipeline - Data Ingestion to vector db pipeline

import os
from pydoc import text
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# read all the pdf files inside the given directory
def process_all_pdf(pdf_directory):
    """process all teh pdf files in a given directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # find all pdf files
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} pdf files in the given directory.")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            document = loader.load()
            
            # adding metadata custome
            for page in document:
                page.metadata['file_name'] = pdf_file.name,
                page.metadata['file_type'] = 'pdf'
                
            all_documents.append(document)
            print(f"\nLoaded {len(document)} pages")
            
        except Exception as e:
            print(f"\nError: {e}")
        
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents
    
# processing all pdf in the directory
all_pdf_documents = process_all_pdf("data/pdf_files")


# Text Splitting - get teh pdf data into chunk
def split_documents(documents, chunk_size = 1000, chunk_overlap = 200):
    '''split documents into smaller chunk for better rag performance'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""]
    )
    splits_docs = text_splitter.split_documents(documents)
    print(f"Splitted {len(documents)} pages into {len(splits_docs)} chunks.")
    return splits_docs

all_pdf_docs = [page for doc in all_pdf_documents for page in doc]
chunks = split_documents(documents = all_pdf_docs)

