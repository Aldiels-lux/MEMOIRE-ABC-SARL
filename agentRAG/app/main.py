from fastapi import FastAPI
from app.schema import QuestionRequest, AnswerResponse
from app.rag.loader import load_and_split_pdfs
from app.rag.vectorstore import get_vectorstore
from app.rag.generator import generate_answer
from app.rag.config import PDF_DIRECTORY

app = FastAPI(title="RAG GPT-2 + BERT + Astra DB")

# Initialisation
docs = load_and_split_pdfs(PDF_DIRECTORY)
vectorstore = get_vectorstore()

# Indexation (une seule fois)
vectorstore.add_documents(docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    retrieved_docs = retriever.get_relevant_documents(request.question)
    context = " ".join(doc.page_content for doc in retrieved_docs)

    answer = generate_answer(request.question, context)

    return AnswerResponse(answer=answer)
