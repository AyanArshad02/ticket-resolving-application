from pdf_rag import query_rag_pdf
from sql.sql_search import query_rag_sql
from langchain_community.chat_models import ChatOpenAI

# Sample query
query = '''WMS dashboard showing incorrect inventory levels for SKU #12345. 
Physical count shows 500 units but system displays 750 units. 
This is affecting our pick operations and causing customer complaints.'''

# Get responses from both sources
pdf_response = query_rag_pdf(query)
sql_response = query_rag_sql(query)  # Should return a list of strings

# Helper: Clean up PDF response content
def clean_response_content(content):
    cleaned = content.strip()
    while cleaned.startswith(('.', ' ', '\n', '\t', '*', '-')):
        cleaned = cleaned[1:].strip()
    return cleaned

# Always show PDF response
print("########### PDF RESPONSE ###########")
pdf_content = pdf_response if isinstance(pdf_response, str) else pdf_response.content
cleaned_pdf_content = clean_response_content(pdf_content)
print(cleaned_pdf_content)

# Always show SQL response
print("########### SQL RESPONSE ###########")
if sql_response:
    print("\n\n---\n\n".join(sql_response))
else:
    print("No results found.")

# Final output decision logic
print("########### FINAL RESPONSE ###########")

if not sql_response or (len(sql_response) == 1 and sql_response[0].lower().startswith("error executing sql:")):
    print("No solution found in previous tickets. Using documentation response:")
    print(cleaned_pdf_content)
else:
    first_result = sql_response[0]
    solution_text = None

    for line in first_result.split('\n'):
        if line.lower().startswith('solution:'):
            solution_text = line[len('solution:'):].strip()
            break

    if solution_text:
        print(solution_text)
    else:
        print(first_result)





# from pdf_rag import query_rag_pdf
# from sql.sql_search import query_rag_sql
# from langchain_community.chat_models import ChatOpenAI

# query = '''WMS dashboard showing incorrect inventory levels for SKU #12345. Physical count shows 500 units but
# system displays 750 units. This is affecting our pick operations and causing customer complaints'''

# # Get responses from both sources
# pdf_response = query_rag_pdf(query)
# sql_response = query_rag_sql(query)  # Returns a list

# # Clean PDF response content
# def clean_response_content(content):
#     """Remove leading whitespace, dots, and other unwanted characters"""
#     cleaned = content.strip()
#     while cleaned.startswith(('.', ' ', '\n', '\t', '*', '-')):
#         cleaned = cleaned[1:].strip()
#     return cleaned

# # Always print PDF response
# print("###########PDF RESPONSE###########")
# cleaned_pdf_content = clean_response_content(pdf_response.content)
# print(cleaned_pdf_content)

# # Always print SQL response (NO "Result 1:" labels)
# print("###########SQL RESPONSE###########")
# if sql_response:
#     # Simply join all results with separator - no "Result X:" labels
#     print("\n\n---\n\n".join(sql_response))
# else:
#     print("No results found.")

# # Determine final response based on SQL results
# print("###########FINAL RESPONSE###########")

# # Check if SQL found any results
# if not sql_response or (len(sql_response) == 1 and sql_response[0].startswith("Error executing SQL:")):
#     # No SQL results found, use PDF response as final
#     print("No solution found in previous tickets. Using documentation response:")
#     print(cleaned_pdf_content)
# else:
#     # SQL results found, extract solution from first result
#     first_result = sql_response[0]
    
#     # Find the line starting with 'solution:'
#     solution_text = None
#     for line in first_result.split('\n'):
#         if line.lower().startswith('solution:'):
#             solution_text = line[len('solution:'):].strip()
#             break
    
#     if solution_text:
#         print(solution_text)
#     else:
#         # Fallback to first result if no solution line found
#         print(first_result)










