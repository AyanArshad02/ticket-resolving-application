# combine_rag.py
from pdf_rag import query_rag_pdf
from sql.sql_search import query_rag_sql
from langchain_community.chat_models import ChatOpenAI

query = "what do you think about tajmahal"

pdf_response = query_rag_pdf(query)
sql_response = query_rag_sql(query)

combined_prompt = f"""
You are a helpful assistant. Below are two responses retrieved from different sources for the same user query.

Response from PDF-based documentation:
{pdf_response}

Response from previously resolved SQL tickets:
{sql_response}

Please merge and rewrite these into a single, clear, concise, and helpful answer for the user.
"""

model = ChatOpenAI()
final_response = model.invoke(combined_prompt)

print("###########PDF RESPONSE###########")
print(pdf_response.content)

print("###########SQL RESPONSE###########")
print(sql_response)

print("###########FINAL RESPONSE###########")
print(final_response.content)

# -------------------------------------------------------------


# from pdf_rag import query_rag_pdf
# from sql.sql_search import query_rag_sql
# from langchain_community.chat_models import ChatOpenAI
# import logging

# def get_combined_response(query: str):
#     try:
#         # Initialize responses
#         pdf_response = None
#         sql_response = None
        
#         # Query PDF with error handling
#         try:
#             pdf_response = query_rag_pdf(query)
#             pdf_content = pdf_response.content if hasattr(pdf_response, 'content') else str(pdf_response)
#         except Exception as e:
#             logging.warning(f"PDF query failed: {e}")
#             pdf_content = "No relevant information found in documentation."
        
#         # Query SQL with error handling
#         try:
#             sql_response = query_rag_sql(query)
#             sql_content = sql_response if isinstance(sql_response, str) else str(sql_response)
#         except Exception as e:
#             logging.warning(f"SQL query failed: {e}")
#             sql_content = "No similar resolved tickets found."
        
#         # Generate combined response
#         final_response = generate_combined_response(query, pdf_content, sql_content)
        
#         return {
#             'pdf_response': pdf_content,
#             'sql_response': sql_content,
#             'final_response': final_response,
#             'success': True
#         }
        
#     except Exception as e:
#         logging.error(f"Error in get_combined_response: {e}")
#         return {
#             'error': str(e),
#             'success': False
#         }

# def generate_combined_response(query: str, pdf_content: str, sql_content: str) -> str:
#     # Check if we have meaningful responses
#     has_pdf_info = pdf_content and "No relevant information" not in pdf_content
#     has_sql_info = sql_content and "No similar resolved tickets" not in sql_content
    
#     if not has_pdf_info and not has_sql_info:
#         return "I couldn't find relevant information in our documentation or previous tickets. Please provide more details or contact support for assistance."
    
#     combined_prompt = f"""
# You are a technical support assistant for Softeon. A user has submitted this query: "{query}"

# I have searched our knowledge base and found the following information:

# Documentation/Manual Information:
# {pdf_content if has_pdf_info else "No relevant documentation found."}

# Previous Ticket Solutions:
# {sql_content if has_sql_info else "No similar resolved tickets found."}

# Instructions:
# 1. If both sources have relevant information, combine them into a comprehensive solution
# 2. If only one source has information, use that and mention the limitation
# 3. Prioritize practical, actionable solutions
# 4. If this appears to be a non-technical query (like about Taj Mahal), politely redirect to technical support topics
# 5. Include ticket references if available from SQL results
# 6. Keep the response professional and helpful

# Provide a clear, structured response that directly addresses the user's query.
# """

#     try:
#         model = ChatOpenAI(temperature=0.3)  # Lower temperature for more consistent responses
#         final_response = model.invoke(combined_prompt)
#         return final_response.content
#     except Exception as e:
#         logging.error(f"Error generating combined response: {e}")
#         return "I encountered an error while processing your request. Please try again or contact support."



# def assess_response_quality(query: str, pdf_response: str, sql_response: str) -> dict:
#     """Assess the quality and relevance of responses"""
#     assessment = {
#         'pdf_relevant': False,
#         'sql_relevant': False,
#         'confidence_score': 0
#     }
    
#     # Simple relevance checking (you can enhance this with more sophisticated methods)
#     query_keywords = set(query.lower().split())
    
#     # Check PDF response relevance
#     if pdf_response and len(pdf_response) > 50:
#         pdf_keywords = set(pdf_response.lower().split())
#         pdf_overlap = len(query_keywords.intersection(pdf_keywords))
#         assessment['pdf_relevant'] = pdf_overlap > 0
    
#     # Check SQL response relevance
#     if sql_response and len(sql_response) > 50:
#         sql_keywords = set(sql_response.lower().split())
#         sql_overlap = len(query_keywords.intersection(sql_keywords))
#         assessment['sql_relevant'] = sql_overlap > 0
    
#     # Calculate confidence score
#     if assessment['pdf_relevant'] and assessment['sql_relevant']:
#         assessment['confidence_score'] = 0.9
#     elif assessment['pdf_relevant'] or assessment['sql_relevant']:
#         assessment['confidence_score'] = 0.6
#     else:
#         assessment['confidence_score'] = 0.2
    
#     return assessment




# def main():
#     query = "WMS dashboard showing incorrect inventory levels"  # More relevant example
    
#     print(f"Processing query: {query}")
#     print("=" * 80)
    
#     # Get combined response
#     result = get_combined_response(query)
    
#     if result['success']:
#         # Assess response quality
#         quality = assess_response_quality(
#             query, 
#             result['pdf_response'], 
#             result['sql_response']
#         )
        
#         print("###########PDF RESPONSE###########")
#         print(result['pdf_response'])
#         print(f"\nPDF Relevance: {'Yes' if quality['pdf_relevant'] else 'No'}")
        
#         print("\n###########SQL RESPONSE###########")
#         print(result['sql_response'])
#         print(f"\nSQL Relevance: {'Yes' if quality['sql_relevant'] else 'No'}")
        
#         print("\n###########FINAL RESPONSE###########")
#         print(result['final_response'])
#         print(f"\nConfidence Score: {quality['confidence_score']:.1f}")
        
#         # Log for analytics
#         log_interaction(query, result, quality)
#     else:
#         print(f"Error processing query: {result['error']}")

# def log_interaction(query: str, result: dict, quality: dict):
#     """Log interaction for analytics and improvement"""
#     log_data = {
#         'timestamp': datetime.now().isoformat(),
#         'query': query,
#         'pdf_relevant': quality['pdf_relevant'],
#         'sql_relevant': quality['sql_relevant'],
#         'confidence_score': quality['confidence_score']
#     }
#     # Save to log file or database for analysis
#     logging.info(f"Interaction logged: {log_data}")


# # # Response Caching
# # import hashlib
# # from functools import lru_cache

# # @lru_cache(maxsize=100)
# # def cached_query_processing(query_hash: str, query: str):
# #     """Cache responses for frequently asked questions"""
# #     return get_combined_response(query)

# # def process_query_with_cache(query: str):
# #     query_hash = hashlib.md5(query.encode()).hexdigest()
# #     return cached_query_processing(query_hash, query)


# # # Response Formatting
# # def format_final_response(response: str) -> str:
# #     """Format the response for better readability"""
# #     if not response:
# #         return "No response generated."
    
# #     # Add structure to the response
# #     formatted = response.replace('\n\n', '\n')
    
# #     # Add sections if not present
# #     if 'Solution:' not in response and 'solution' in response.lower():
# #         formatted = formatted.replace('solution', '\n**Solution:**', 1)
    
# #     return formatted











