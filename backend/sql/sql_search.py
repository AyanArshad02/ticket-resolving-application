from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Path to your SQLite DB
DB_PATH = "/Users/mdayanarshad/Desktop/Softeon/ticket-resolving-application/backend/ticket.db"

# Prompt Template
sql_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert data analyst. Generate a SQLite query to find semantically similar tickets based on the user's question.

Database Schema:
Table: user_ticket
Columns: ticket_id, customer_name, description, category, priority, status, resolved_by_team, solution, created_date, resolved_date, time_to_resolve_hours

User Question: {question}

SQL Generation Rules:
1. ALWAYS search primarily in the 'description' field for similarity
2. Extract key concepts and technical terms from the question
3. Use OR logic to find similar issues, not exact matches
4. Ignore specific identifiers (SKU numbers, exact counts, IDs) when looking for similar patterns
5. Focus on problem types, symptoms, and affected systems

Query Pattern Guidelines:
- For technical issues: Look for system names, error types, symptoms
- For inventory issues: Focus on WMS, inventory, stock, count, levels, dashboard terms
- For integration issues: Look for sync, API, connection, interface terms  
- For performance issues: Look for slow, timeout, delay, performance terms
- For user issues: Look for login, access, permission, authentication terms

Construction Strategy:
1. Identify the main problem domain from the question
2. Extract 3-5 key terms that represent the core issue
3. Create OR conditions that capture variations of the same concept
4. Use parentheses to group related concepts with AND, then OR different concept groups

Return only the SQL SELECT statement. No explanations.
"""
)

# Executes SQL and returns result
def run_sql(query):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        return columns, rows
    except sqlite3.Error as e:
        raise Exception(f"SQLite error: {e}")

# Main RAG-like function for SQL
def query_rag_sql(nl_query):
    llm = ChatOpenAI(temperature=0.2)
    
    # Generate SQL using LLM
    try:
        sql_query = llm.invoke(sql_prompt.format(question=nl_query)).content.strip()
    except Exception as e:
        return [f"LLM error: {str(e)}"]

    # Clean formatting (remove ```sql blocks)
    if "```" in sql_query:
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    sql_query = sql_query.rstrip(';').strip()
    print("Generated SQL:\n", sql_query)  # Optional debug

    # Execute query
    try:
        columns, rows = run_sql(sql_query)
        if not rows:
            return []

        results = []
        for row in rows:
            filtered = {
                col: val for col, val in zip(columns, row)
                if col.lower() not in ['index', 'rowid']
            }
            results.append("\n".join([f"{k}: {v}" for k, v in filtered.items()]))

        return results

    except Exception as e:
        return [f"Error executing SQL: {str(e)}"]

# Test Queries
def test_different_queries():
    test_queries = [
        "WMS dashboard showing incorrect inventory levels for SKU #12345",
        "User cannot login to the system",
        "Application is running very slow",
        "API integration failing with timeout errors",
        "Database connection issues",
        "Email notifications not being sent",
        "Report generation taking too long"
    ]
    
    print("Running Query Tests\n" + "=" * 50)
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = query_rag_sql(query)
        print(f"Results found: {len(results)}")
        for r in results:
            print(r)
        print("-" * 30)










# from langchain_community.chat_models import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# import sqlite3
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Path to your SQLite DB
# db_path = "/Users/mdayanarshad/Desktop/Softeon/ticket-resolving-application/backend/ticket.db"

# # Universal SQL generation prompt that works for all query types
# sql_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""
# You are an expert data analyst. Generate a SQLite query to find semantically similar tickets based on the user's question.

# Database Schema:
# Table: ticket
# Columns: ticket_id, customer_name, description, category, priority, status, resolved_by_team, solution, created_date, resolved_date, time_to_resolve_hours

# User Question: {question}

# SQL Generation Rules:
# 1. ALWAYS search primarily in the 'description' field for similarity
# 2. Extract key concepts and technical terms from the question
# 3. Use OR logic to find similar issues, not exact matches
# 4. Ignore specific identifiers (SKU numbers, exact counts, IDs) when looking for similar patterns
# 5. Focus on problem types, symptoms, and affected systems

# Query Pattern Guidelines:
# - For technical issues: Look for system names, error types, symptoms
# - For inventory issues: Focus on WMS, inventory, stock, count, levels, dashboard terms
# - For integration issues: Look for sync, API, connection, interface terms  
# - For performance issues: Look for slow, timeout, delay, performance terms
# - For user issues: Look for login, access, permission, authentication terms

# Construction Strategy:
# 1. Identify the main problem domain from the question
# 2. Extract 3-5 key terms that represent the core issue
# 3. Create OR conditions that capture variations of the same concept
# 4. Use parentheses to group related concepts with AND, then OR different concept groups

# Example patterns:
# - Inventory: WHERE (description LIKE '%WMS%' AND description LIKE '%inventory%') OR (description LIKE '%stock%' AND description LIKE '%count%') OR (description LIKE '%dashboard%' AND description LIKE '%levels%')
# - Login: WHERE (description LIKE '%login%' OR description LIKE '%authentication%') AND (description LIKE '%failed%' OR description LIKE '%error%' OR description LIKE '%cannot%')
# - Performance: WHERE (description LIKE '%slow%' OR description LIKE '%timeout%' OR description LIKE '%delay%') AND (description LIKE '%system%' OR description LIKE '%application%' OR description LIKE '%response%')

# Return only the SQL SELECT statement. No explanations.
# """
# )

# # Function to run SQL on SQLite
# def run_sql(query):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute(query)
#     rows = cursor.fetchall()
#     col_names = [description[0] for description in cursor.description]
#     conn.close()
#     return col_names, rows

# # Main function to handle natural language query -> SQL -> Result
# def query_rag_sql(nl_query):
#     llm = ChatOpenAI(temperature=0.2)  # Even lower temperature for consistency
    
#     # Step 1: Generate SQL from NL query
#     sql_query = llm.invoke(sql_prompt.format(question=nl_query)).content.strip()

#     # Step 2: Clean up SQL formatting
#     # Remove markdown code blocks
#     if "```sql" in sql_query:
#         sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
#     elif "```" in sql_query:
#         sql_query = sql_query.replace("```", "").strip()
    
#     # Remove any trailing semicolons or extra whitespace
#     sql_query = sql_query.rstrip(';').strip()

#     # Optional: Print for debugging
#     print("Generated SQL:\n", sql_query)

#     # Step 3: Execute and format results
#     try:
#         columns, result = run_sql(sql_query)

#         if result:
#             formatted_rows = []
#             for row in result:
#                 row_dict = dict(zip(columns, row))

#                 # Filter out unwanted columns
#                 filtered_items = [(col, val) for col, val in row_dict.items()
#                                   if col.lower() not in ['index', 'rowid']]

#                 row_str = "\n".join([f"{col}: {val}" for col, val in filtered_items])
#                 formatted_rows.append(row_str)

#             return formatted_rows  # Returns LIST of strings
#         else:
#             return []  # No results

#     except Exception as e:
#         return [f"Error executing SQL: {str(e)}"]  # Return error in list

# # Optional: Test function to validate different query types
# def test_different_queries():
#     """Test function to validate the prompt works for various query types"""
#     test_queries = [
#         "WMS dashboard showing incorrect inventory levels for SKU #12345",
#         "User cannot login to the system",
#         "Application is running very slow",
#         "API integration failing with timeout errors",
#         "Database connection issues",
#         "Email notifications not being sent",
#         "Report generation taking too long"
#     ]
    
#     print("Testing different query types:")
#     print("=" * 50)
    
#     for query in test_queries:
#         print(f"\nQuery: {query}")
#         results = query_rag_sql(query)
#         print(f"Results found: {len(results)}")
#         print("-" * 30)


 
