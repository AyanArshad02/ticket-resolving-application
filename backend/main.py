from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_rag import query_rag_pdf
from sql.sql_search import query_rag_sql
from langchain_community.chat_models import ChatOpenAI


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryInput(BaseModel):
    query: str


@app.post("/query")
async def get_responses(data: QueryInput):
    query = data.query


    try:
        # Get PDF-based and SQL-based responses
        pdf_response = query_rag_pdf(query)
        sql_response = query_rag_sql(query)


        # Combine them into one prompt
        combined_prompt = f"""
        You are a helpful assistant. Below are two responses retrieved from different sources for the same user query.


        Response from PDF-based documentation:
        {pdf_response.content if hasattr(pdf_response, "content") else pdf_response}


        Response from previously resolved SQL tickets:
        {sql_response}


        Please merge and rewrite these into a single, clear, concise, and helpful answer for the user.
        """


        model = ChatOpenAI()
        final_response = model.invoke(combined_prompt)


        # Optional logging for debugging
        print("###########PDF RESPONSE###########")
        print(pdf_response.content if hasattr(pdf_response, "content") else pdf_response)
        print("###########SQL RESPONSE###########")
        print(sql_response)
        print("###########FINAL RESPONSE###########")
        print(final_response.content)


        return {
            "pdf_response": pdf_response.content if hasattr(pdf_response, "content") else pdf_response,
            "sql_response": sql_response,
            "final_response": final_response.content
        }


    except Exception as e:
        return {
            "pdf_response": f"Error: {str(e)}",
            "sql_response": f"Error: {str(e)}",
            "final_response": f"Sorry, an error occurred: {str(e)}"
        }


@app.get("/")
async def root():
    return {"message": "Support System RAG API is running"}
