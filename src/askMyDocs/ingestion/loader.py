from askMyDocs.ingestion.ingest import collect_documents
from langchain_community.document_loaders import PyPDFLoader


def load_documents():
    files = collect_documents()

    documents = []

    for file in files:
        loader = PyPDFLoader(str(file))
        documents.extend(loader.load())

    return documents
