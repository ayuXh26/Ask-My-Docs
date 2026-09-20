from src.askMyDocs.retrieval.retriever import create_retriever
from src.askMyDocs.generation.context_builder import build_context

retriever = create_retriever()

documents = retriever.invoke(
    "Explain how a URL shortener works."
)

context, sources = build_context(documents)

print("===== CONTEXT =====")
print(context)

print("\n===== SOURCES =====")
print(sources)