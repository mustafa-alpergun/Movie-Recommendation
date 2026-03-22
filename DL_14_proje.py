import pandas as pd
import os
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
warnings.filterwarnings('ignore')

file_path = r"C:\Users\muham\Downloads\archive (12)\movies_metadata.csv"
df = pd.read_csv(file_path, low_memory=False)

columns_to_keep = ['title', 'overview', 'genres', 'release_date', 'vote_average']
df_movies = df[columns_to_keep].copy()
df_movies = df_movies.dropna(subset=['overview', 'title'])
df_sample = df_movies.head(1000).copy()

print("Loading data and generating embeddings (Please wait)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
overviews = df_sample['overview'].tolist()
embeddings = model.encode(overviews, show_progress_bar=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable!")

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash') 

print("\n" + "="*60)
print("🎬 WELCOME TO THE MOVIE ASSISTANT! 🎬")
print("Type 'q', 'quit', or 'exit' to leave.")
print("="*60)

while True:
    user_query = input("\nWhat kind of movie would you like to watch? \n👉 You: ")
    
    if user_query.lower() in ['q', 'quit', 'exit']:
        print("\nSee you later! Enjoy your movie... 🍿")
        break
        
    if not user_query.strip():
        continue

    print("🔍 Searching the database...")
    
    query_vector = model.encode([user_query])
    similarity_scores = cosine_similarity(query_vector, embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:3]

    found_movies_text = ""
    for i in top_indices:
        movie_title = df_sample.iloc[i]['title']
        movie_overview = df_sample.iloc[i]['overview']
        found_movies_text += f"- Movie: {movie_title}\n  Overview: {movie_overview}\n\n"

    prompt = f"""
    You are a highly knowledgeable, fun, and movie-loving AI assistant. 
    The user has asked you for a movie recommendation. 
    Below are the most suitable movies and their overviews that I found for you from my own database. 
    PLEASE USE ONLY THESE MOVIES to write a natural, conversational, and recommendatory response to the user. Do not invent or suggest any outside movies.

    User's Request: {user_query}

    Movies Found from Database (Context):
    {found_movies_text}
    """

    print(" Gemini is preparing a response...")
    try:
        response = model_gemini.generate_content(prompt)
        print("\n--- 🎬 ASSISTANT'S RESPONSE ---")
        print(response.text)
        print("-" * 60)
    except Exception as e:
        print(f"\n An error occurred: {e}")
