from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from pymupdf import Document

class ProcessDocumentsManager():
    # read all the pdf files inside the given directory
    def process_all_documents(
        self,
        documents_directory: str, 
        file_type: str
        ) -> list[Document]:
        """process all the files in a given directory"""
        all_documents = []
        pdf_dir = Path(documents_directory)

        files = []
        
        if file_type == "pdf":
            # find all pdf files
            pdf_files = list(pdf_dir.glob("**/*.pdf"))
            print(f"\nFound {len(pdf_files)} pdf files in the given directory.")
            files.extend(pdf_files)
        else:
            raise ValueError("\nInvalid or missing file_type. Only 'pdf' is supported")

        for file_ in files:
            print(f"\nProcessing: {file_.name}")
            try:
                if file_type == "pdf":
                    loader = PyMuPDFLoader(str(file_))
                document = loader.load()

                # adding metadata custome
                for page in document:
                    page.metadata["file_name"] = (file_.name,)
                    page.metadata["file_type"] = "pdf"

                all_documents.append(document)
                print(f"\nLoaded {len(document)} pages")

            except Exception as e:
                print(f"\nError: {e}")

        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents
