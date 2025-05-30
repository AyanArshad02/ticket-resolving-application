import os
import sqlite3
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
 
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
# Database location
db_path = "ticket.db"


# Define template to convert natural language to SQL
sql_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert data analyst. Use the schema below to generate an SQL query for SQLite.
 
Schema:
Table: ticket
Columns: ticket_id,customer_name,description,category,priority,status,resolved_by_team,solution,created_date,resolved_date,time_to_resolve_hours
 
Now, convert the following natural language question into a SQL query:
Question: {question}
 
Only return the SQL query. Do not explain anything.
"""
)
 
# Function to run SQL query
def run_sql(query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    col_names = [description[0] for description in cursor.description]
    conn.close()
    return col_names, rows

 
# Main function to handle NL2SQL and query execution
def query_rag_sql(nl_query):
    llm = ChatOpenAI(temperature=0.5)
    sql_query = llm.invoke(sql_prompt.format(question=nl_query)).content.strip()
   
    try:
        columns, result = run_sql(sql_query)
       
        # Try to extract only the 'solution' column
        if "solution" in columns:
            solution_index = columns.index("solution")
            solutions = [row[solution_index] for row in result]
            solutions_text = "\n\n".join(solutions) if solutions else "No solutions found."
            return f"{solutions_text}"
        else:
            return f"SQL: {sql_query}\n\nError: 'solution' column not found in result."
       
    except Exception as e:
        return f"SQL: {sql_query}\n\nError executing SQL: {str(e)}"



 
