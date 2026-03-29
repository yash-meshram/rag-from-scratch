# Data Ingestion

import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyMuPDFLoader,
)

# Document Structure
doc = Document(
    page_content="here is teh information about Traditional RAG",
    metadata={
        "source": "testfile.txt",
        "page": 1,
        "author": "yash",
        "created_date": "01-01-2026",
    },
)
os.makedirs("data/text_files", exist_ok=True)

sample_texts = {
    "data/text_files/python_intro.txt": """Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",
    "data/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems""",
}

for filepath, content in sample_texts.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
print("Sample files created")

# TextLoader
loader = TextLoader("data/text_files/python_intro.txt", encoding="utf-8")
document = loader.load()

# Directory Loader
# load all the text file from the directory
dir_loader = DirectoryLoader(
    "data/text_files",
    glob="**/*.txt",  # pattern to match the file
    loader_cls=TextLoader,  # loader class to use
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
)
documents = dir_loader.load()

# pdf loader
dir_loader = DirectoryLoader(
    "data/pdf_files",
    glob="**/*.pdf",  # pattern to match the file
    loader_cls=PyMuPDFLoader,  # loader class to use
    show_progress=True,
)
pdf_documents = dir_loader.load()

type(pdf_documents)
type(pdf_documents[0])
