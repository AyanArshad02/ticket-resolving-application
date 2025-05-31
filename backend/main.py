from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_rag import query_rag_pdf
from sql.sql_search import query_rag_sql
from langchain_community.chat_models import ChatOpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Just for development; restrict in production
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
        pdf_response = query_rag_pdf(query)
        sql_response = query_rag_sql(query)

        def clean_response_content(content):
            cleaned = content.strip()
            while cleaned.startswith(('.', ' ', '\n', '\t', '*', '-')):
                cleaned = cleaned[1:].strip()
            return cleaned

        cleaned_pdf_content = clean_response_content(pdf_response)

        print("###########PDF RESPONSE###########")
        print(cleaned_pdf_content)

        print("###########SQL RESPONSE###########")
        if sql_response:
            print("\n\n---\n\n".join(sql_response))
        else:
            print("No results found.")

        print("###########FINAL RESPONSE###########")
        if not sql_response or (len(sql_response) == 1 and sql_response[0].startswith("Error executing SQL:")):
            final_result = f"No solution found in previous tickets. Using documentation response:\n{cleaned_pdf_content}"
        else:
            first_result = sql_response[0]
            solution_text = None
            for line in first_result.split('\n'):
                if line.lower().startswith('solution:'):
                    solution_text = line[len('solution:'):].strip()
                    break
            final_result = solution_text if solution_text else first_result

        return {
            "pdf_response": cleaned_pdf_content,
            "sql_response": sql_response,
            "final_response": final_result
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
