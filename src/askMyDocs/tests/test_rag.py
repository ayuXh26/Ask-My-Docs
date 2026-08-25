from src.askMyDocs.rag.rag import create_rag_chain

rag_chain = create_rag_chain()

question = "What is Cricket World Cup 2023"

answer = rag_chain(question)

print(answer)