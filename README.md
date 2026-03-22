Uzun süredir üzerinde heyecanla uğraştığım ve nihayet son halini verdiğim yeni yapay zeka projemi sizlerle paylaşmaktan mutluluk duyuyorum! 🍿🎬 Makine öğrenmesi algoritmalarını ve Büyük Dil Modellerini (LLM) günlük hayatta kullanılabilecek eğlenceli bir ürüne dönüştürmek her zaman en sevdiğim şeylerden biri olmuştur. 👇

🔍 Proje: RAG Mimarisi ile Akıllı Sinema Asistanı

Bu projede, kullanıcının ruh haline veya izlemek istediği konuya göre ("Karlar altında geçen gerilimli bir cinayet filmi" gibi) film tavsiyeleri veren bir RAG (Retrieval-Augmented Generation) uygulaması geliştirdim.

Sistem nasıl çalışıyor?
1️⃣ Önce kullanıcının cümlesi Sentence Transformers (all-MiniLM-L6-v2) kullanılarak vektörlere ayrılıyor.
2️⃣ Kendi yerel film veritabanımdaki özetler arasında Cosine Similarity (Kosinüs Benzerliği) ile anlamsal bir arama yapılarak en uygun 3 film bulunuyor.
3️⃣ Bulunan bu ham veri, Gemini 2.5 Flash modeline bağlam (context) olarak sunuluyor ve yapay zeka bu filmleri doğal, sohbet havasında bir sinema eleştirmeni gibi kullanıcıya yorumluyor.

Dışarıdan halüsinasyon (hallucination) ile film uydurmasını engellemek için modeli sadece kendi sağladığım veritabanıyla sınırlandırdım.

💻 Kendi bilgisayarında denemek istersen:
Projeyi GitHub'a yükledim. Kodu alıp kendi API anahtarınla hemen test edebilirsin. Gerekli kütüphaneleri kurduktan sonra terminale şu komutu yazman yeterli:

python -m streamlit run app.py

🛠️ Kullanılan Teknolojiler: Python, Streamlit, Google Gemini API, Sentence Transformers, Scikit-Learn, Pandas, NumPy

Görüşlerinizi bekliyorum! 🚀

veri : https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

🇬🇧 I am thrilled to share my new AI project, which I have been working on for quite a while and have finally completed! 🍿🎬 Transforming machine learning algorithms and Large Language Models (LLMs) into fun, everyday products is always a great experience. 👇

🔍 Project: Smart Cinema Assistant powered by RAG

I built a RAG (Retrieval-Augmented Generation) application that gives personalized movie recommendations based on user prompts (e.g., "A tense murder mystery set in the snow").

How it works:
1️⃣ The user's query is converted into embeddings using Sentence Transformers (all-MiniLM-L6-v2).
2️⃣ The system performs a semantic search in a local movie database using Cosine Similarity to retrieve the top 3 matching films.
3️⃣ This retrieved context is fed into the Gemini 2.5 Flash model, which then acts as a conversational film critic, recommending these specific movies to the user.

To prevent hallucinations, the LLM is strictly prompted to generate responses based only on the provided local database context.

💻 Want to try it yourself?
You can grab the code from my GitHub, insert your Gemini API key, and run it locally. Just install the dependencies and type this into your terminal:

python -m streamlit run app.py

🛠️ Tech Stack: Python, Streamlit, Google Gemini API, Sentence Transformers, Scikit-Learn, Pandas, NumPy

I would love to hear your feedback! 🚀

data : https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
