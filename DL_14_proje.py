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

print("Veriler yükleniyor ve Embedding işlemi yapılıyor (Lütfen bekleyin)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
ozetler = df_sample['overview'].tolist()
embeddings = model.encode(ozetler, show_progress_bar=True)

GEMINI_API_KEY = "google apı key girin" 
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.5-flash')

print("\n" + "="*60)
print(" SİNEMA ASİSTANINA HOŞ GELDİN! ")
print("Çıkmak için 'q', 'çıkış' veya 'exit' yazabilirsin.")
print("="*60)



while True:
   
    kullanici_sorusu = input("\nNe tür bir film izlemek istersin? \n👉 Sen: ")
    
    
    if kullanici_sorusu.lower() in ['q', 'çık', 'çıkış', 'exit']:
        print("\nGörüşmek üzere! İyi seyirler... ")
        break
        
    
    if not kullanici_sorusu.strip():
        continue

    print(" Veritabanında aranıyor...")
    
    
    soru_vektoru = model.encode([kullanici_sorusu])
    benzerlik_skorlari = cosine_similarity(soru_vektoru, embeddings)[0]
    en_benzer_indeksler = np.argsort(benzerlik_skorlari)[::-1][:3]

    bulunan_filmler_metni = ""
    for i in en_benzer_indeksler:
        film_adi = df_sample.iloc[i]['title']
        ozet = df_sample.iloc[i]['overview']
        bulunan_filmler_metni += f"- Film: {film_adi}\n  Özet: {ozet}\n\n"

    prompt = f"""
    Sen çok bilgili, eğlenceli ve sinema aşığı bir yapay zeka asistanısın. 
    Kullanıcı sana bir film tavsiyesi sordu. 
    Aşağıda senin için kendi veritabanımdan bulduğum en uygun filmler ve özetleri var. 
    LÜTFEN SADECE BU FİLMLERİ KULLANARAK kullanıcıya doğal, sohbet havasında ve tavsiye niteliğinde bir cevap yaz. Dışarıdan başka film uydurma.

    Kullanıcının İsteği: {kullanici_sorusu}

    Veritabanından Bulunan Filmler (Bağlam):
    {bulunan_filmler_metni}
    """

    print(" Gemini yorumunu hazırlıyor...")
    try:
        response = model_gemini.generate_content(prompt)
        print("\n---  ASİSTANIN CEVABI ---")
        print(response.text)
        print("-" * 60)
    except Exception as e:
        print(f"\n Bir hata oluştu: {e}")
