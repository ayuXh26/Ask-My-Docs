from src.askMyDocs.ingestion.ingest import collect_documents

files = collect_documents()

for file in files:
    print(file)