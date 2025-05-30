from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Paths for PDFs and vector store
dir_path = "/Users/mdayanarshad/Desktop/Softeon/ticket-resolving-application/backend/data"
chroma_path = "/Users/mdayanarshad/Desktop/Softeon/ticket-resolving-application/backend/chroma"

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load and process documents
def load_documents(dir_path):
    loader = DirectoryLoader(
        path=dir_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    return docs

def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    return chunks

def save_to_chroma(chunks):
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=chroma_path
    )
    db.persist()

# Optional: Only save data if DB doesn't exist
def save_data():
    if not os.path.exists(os.path.join(chroma_path, "index")):
        documents = load_documents(dir_path)
        chunks = split_text(documents)
        save_to_chroma(chunks)

save_data()

# Main function for querying PDF knowledge base
def query_rag_pdf(query):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    docs_with_scores = db.similarity_search_with_relevance_scores(query, k=3)

    if not docs_with_scores:
        return "No relevant documents found."

    # Build context from retrieved documents
    context = "\n\n".join([doc.page_content for doc, score in docs_with_scores])

    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

    model = ChatOpenAI(temperature=0.2)
    response = model.invoke(prompt)

    return response.content.strip()

# Test
if __name__ == "__main__":
    test_query = "What is the resolution process for failed API integrations?"
    print(query_rag_pdf(test_query))









# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import Document
# from dotenv import load_dotenv
# import os

# dir_path="/Users/mdayanarshad/Desktop/Softeon/ticket-resolving-application/backend/data"
# chroma_path="/Users/mdayanarshad/Desktop/Softeon/ticket-resolving-application/backend/chroma"

# load_dotenv()

# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# def load_documents(dir_path):
#     loader=DirectoryLoader(
#         path=dir_path,
#         glob="**/*.pdf",
#         loader_cls=PyPDFLoader
#         )
#     docs=loader.load()
#     return docs

# def split_text(docs):
#     splitter=RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks=splitter.split_documents(docs)
#     return chunks

# def save_to_chroma(chunks):
#     db=Chroma.from_documents(
#         chunks,
#         OpenAIEmbeddings(),
#         persist_directory=chroma_path
#     )
#     db.persist()

# def save_data():
#     documents=load_documents(dir_path)
#     chunks=split_text(documents)
#     save_to_chroma(chunks)

# save_data()

# def query_rag_pdf(query):
#     embedding_function=OpenAIEmbeddings()
#     db=Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
#     results=db.similarity_search_with_relevance_scores(query,k=3)
#     model=ChatOpenAI()
#     results=model.invoke(query)
#     return results

'''query="WMS dashboard showing incorrect inventory levels for SKU #12345."

results=query_rag_pdf(query)
print(results.content)'''