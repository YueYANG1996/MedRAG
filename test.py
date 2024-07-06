from src.utils import RetrievalSystem

retrieval_system = RetrievalSystem(retriever_name="BM25", corpus_name="PubMed_all")

# write a chatbot
while True:
    question = input("Ask me a question: (type 'exit' to quit)\n"
    if question == "exit":
        break

    retrieved_snippets, scores = retrieval_system.retrieve(question, k=5, rrf_k=100)

    for i, snippet in enumerate(retrieved_snippets):
        print(f"{i+1}. ID: {snippet['id']}, Title: {snippet['title']}, \n {snippet['contents']} \n\n\n")