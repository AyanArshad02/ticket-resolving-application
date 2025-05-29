import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Tuple, Dict
import re

# Alternative: Using sentence-transformers for better semantic understanding
# Uncomment the lines below if you want to use sentence-transformers
# from sentence_transformers import SentenceTransformer
# import torch

class TicketSemanticSearch:
    def __init__(self, db_path: str, model_type: str = 'tfidf'):
        """
        Initialize the semantic search system
        
        Args:
            db_path: Path to SQLite database
            model_type: 'tfidf' or 'sentence_transformer'
        """
        self.db_path = db_path
        self.model_type = model_type
        self.vectorizer = None
        self.embeddings = None
        self.tickets_df = None
        
        # Initialize the embedding model
        if model_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
        elif model_type == 'sentence_transformer':
            # Uncomment if using sentence-transformers
            # self.model = SentenceTransformer('all-MiniLM-L6-v2')
            pass
        
        # Load and prepare data
        self.load_tickets()
        
    def load_tickets(self):
        """Load tickets from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Fetch all ticket data
            query = """
            SELECT ticket_id, customer_name, description, category, priority, 
                   status, resolved_by_team, solution, created_date, resolved_date, 
                   time_to_resolve_hours
            FROM user_ticket
            """
            
            self.tickets_df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"Loaded {len(self.tickets_df)} tickets from database")
            
        except Exception as e:
            print(f"Error loading tickets: {e}")
            return None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better embeddings"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def create_embeddings(self, save_embeddings: bool = True):
        """Create embeddings for all ticket descriptions"""
        
        if self.tickets_df is None or self.tickets_df.empty:
            print("No tickets loaded. Please load tickets first.")
            return
        
        # Preprocess descriptions
        descriptions = self.tickets_df['description'].apply(self.preprocess_text).tolist()
        
        if self.model_type == 'tfidf':
            print("Creating TF-IDF embeddings...")
            self.embeddings = self.vectorizer.fit_transform(descriptions)
            
        elif self.model_type == 'sentence_transformer':
            print("Creating Sentence Transformer embeddings...")
            # Uncomment if using sentence-transformers
            # self.embeddings = self.model.encode(descriptions, convert_to_tensor=False)
            pass
        
        if save_embeddings:
            self.save_embeddings()
        
        print(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def save_embeddings(self):
        """Save embeddings and vectorizer to disk"""
        try:
            if self.model_type == 'tfidf':
                # Save TF-IDF vectorizer and embeddings
                with open('tfidf_vectorizer.pkl', 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                
                with open('tfidf_embeddings.pkl', 'wb') as f:
                    pickle.dump(self.embeddings, f)
                    
            elif self.model_type == 'sentence_transformer':
                # Save sentence transformer embeddings
                np.save('sentence_embeddings.npy', self.embeddings)
            
            print("Embeddings saved successfully!")
            
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    def load_embeddings(self):
        """Load pre-computed embeddings from disk"""
        try:
            if self.model_type == 'tfidf':
                with open('tfidf_vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open('tfidf_embeddings.pkl', 'rb') as f:
                    self.embeddings = pickle.load(f)
                    
            elif self.model_type == 'sentence_transformer':
                self.embeddings = np.load('sentence_embeddings.npy')
            
            print("Embeddings loaded successfully!")
            return True
            
        except FileNotFoundError:
            print("No saved embeddings found. Creating new ones...")
            return False
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def search_similar_tickets(self, new_ticket_description: str, top_k: int = 5) -> List[Dict]:
        """
        Find most similar tickets based on description
        
        Args:
            new_ticket_description: Description of the new ticket
            top_k: Number of similar tickets to return
            
        Returns:
            List of dictionaries containing similar tickets and their solutions
        """
        
        if self.embeddings is None:
            if not self.load_embeddings():
                print("Creating embeddings first...")
                self.create_embeddings()
        
        # Preprocess the new ticket description
        processed_description = self.preprocess_text(new_ticket_description)
        
        if self.model_type == 'tfidf':
            # Transform new ticket description using existing vectorizer
            new_ticket_embedding = self.vectorizer.transform([processed_description])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(new_ticket_embedding, self.embeddings).flatten()
            
        elif self.model_type == 'sentence_transformer':
            # Create embedding for new ticket
            # new_ticket_embedding = self.model.encode([processed_description])
            # similarities = cosine_similarity(new_ticket_embedding, self.embeddings).flatten()
            pass
        
        # Get top K similar tickets
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            ticket_info = {
                'rank': i + 1,
                'similarity_score': similarities[idx],
                'ticket_id': self.tickets_df.iloc[idx]['ticket_id'],
                'customer_name': self.tickets_df.iloc[idx]['customer_name'],
                'description': self.tickets_df.iloc[idx]['description'],
                'category': self.tickets_df.iloc[idx]['category'],
                'priority': self.tickets_df.iloc[idx]['priority'],
                'resolved_by_team': self.tickets_df.iloc[idx]['resolved_by_team'],
                'solution': self.tickets_df.iloc[idx]['solution'],
                'time_to_resolve_hours': self.tickets_df.iloc[idx]['time_to_resolve_hours']
            }
            results.append(ticket_info)
        
        return results
    
    def get_solution_recommendation(self, new_ticket_description: str) -> Dict:
        """
        Get the best solution recommendation for a new ticket
        
        Args:
            new_ticket_description: Description of the new ticket
            
        Returns:
            Dictionary with the most relevant solution and confidence score
        """
        
        similar_tickets = self.search_similar_tickets(new_ticket_description, top_k=1)
        
        if not similar_tickets:
            return {"error": "No similar tickets found"}
        
        best_match = similar_tickets[0]
        
        recommendation = {
            'confidence_score': best_match['similarity_score'],
            'recommended_solution': best_match['solution'],
            'similar_ticket_id': best_match['ticket_id'],
            'similar_category': best_match['category'],
            'similar_priority': best_match['priority'],
            'estimated_resolution_time': best_match['time_to_resolve_hours'],
            'recommended_team': best_match['resolved_by_team']
        }
        
        return recommendation

# Example usage and testing
def main():
    # Initialize the semantic search system
    db_path = "ticket.db"  # Replace with your actual database path
    search_system = TicketSemanticSearch(db_path, model_type='tfidf')
    
    # Create embeddings (this only needs to be done once)
    search_system.create_embeddings()
    
    # Example: Search for similar tickets
    new_ticket = """
    Inventory levels showing incorrect counts in the dashboard. 
    Physical count doesn't match system display causing operational issues.
    """
    
    print("=== Searching for Similar Tickets ===")
    similar_tickets = search_system.search_similar_tickets(new_ticket, top_k=3)
    
    for ticket in similar_tickets:
        print(f"\nRank: {ticket['rank']}")
        print(f"Similarity Score: {ticket['similarity_score']:.4f}")
        print(f"Ticket ID: {ticket['ticket_id']}")
        print(f"Category: {ticket['category']}")
        print(f"Description: {ticket['description'][:100]}...")
        print(f"Solution: {ticket['solution'][:100]}...")
        print("-" * 80)
    
    print("\n=== Solution Recommendation ===")
    recommendation = search_system.get_solution_recommendation(new_ticket)
    
    print(f"Confidence Score: {recommendation['confidence_score']:.4f}")
    print(f"Recommended Team: {recommendation['recommended_team']}")
    print(f"Category: {recommendation['similar_category']}")
    print(f"Estimated Resolution Time: {recommendation['estimated_resolution_time']} hours")
    print(f"\nRecommended Solution:")
    print(recommendation['recommended_solution'])

# Additional utility functions
def batch_search_tickets(search_system, new_tickets: List[str]) -> List[Dict]:
    """Process multiple tickets at once"""
    results = []
    for i, ticket_desc in enumerate(new_tickets):
        print(f"Processing ticket {i+1}/{len(new_tickets)}...")
        recommendation = search_system.get_solution_recommendation(ticket_desc)
        results.append({
            'input_description': ticket_desc,
            'recommendation': recommendation
        })
    return results

def export_results_to_csv(results: List[Dict], filename: str = 'ticket_recommendations.csv'):
    """Export search results to CSV"""
    df_results = pd.DataFrame(results)
    df_results.to_csv(filename, index=False)
    print(f"Results exported to {filename}")

if __name__ == "__main__":
    main()