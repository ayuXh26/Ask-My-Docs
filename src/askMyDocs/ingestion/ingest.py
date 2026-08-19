from pathlib import Path

def collect_documents():
    base_dir = Path(__file__).resolve().parent.parent # -> this gives me the src folder
    docs_dir = base_dir / "data" / "documents" / "langgraph"

    files = []

    for file in docs_dir.rglob("*.pdf"):
        if  file.is_file():
            files.append(file)

    return files

