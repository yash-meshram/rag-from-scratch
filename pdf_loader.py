# RAG pipeline - Data Ingestion to vector db pipeline

import os
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
            documents = loader.load()
            
            # adding metadata custome
            for doc in documents:
                doc.metadata['file_name'] = pdf_file.name,
                doc.metadata['file_type'] = 'pdf'
                
            all_documents.append(documents)
            print(f"\nLoaded {len(documents)} pages")
            
        except Exception as e:
            print(f"\nError: {e}")
        
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents
    
# processing all pdf in the directory
all_pdf_documents = process_all_pdf("data/pdf_files")


# Text Splitting - get teh pdf data into chunk




        