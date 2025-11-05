"""
🔄 Data Preprocessing - Pembersihan dan Persiapan Data
Script untuk melakukan preprocessing pada data teks sebelum training model
"""

import pandas as pd
import re
import pickle
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings

warnings.filterwarnings('ignore')

class TextPreprocessor:
    """Class untuk preprocessing teks bahasa Indonesia"""
    
    def __init__(self):
        # Initialize Sastrawi untuk stopword removal dan stemming
        self.stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = self.stopword_factory.create_stop_word_remover()
        
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()
    
    def clean_text(self, text):
        """
        Pembersihan teks dasar:
        - Hilangkan URL
        - Hilangkan mention (@username)
        - Hilangkan hashtag (#)
        - Hilangkan angka
        - Hilangkan tanda baca berlebih
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Hilangkan URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Hilangkan mention
        text = re.sub(r'@\w+', '', text)
        
        # Hilangkan hashtag (tapi simpan teksnya)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Hilangkan angka
        text = re.sub(r'\d+', '', text)
        
        # Hilangkan tanda baca berlebih
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Hilangkan spasi berlebih
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def case_folding(self, text):
        """Ubah semua huruf menjadi lowercase"""
        return text.lower()
    
    def remove_stopwords(self, text):
        """Hapus stopword menggunakan Sastrawi"""
        return self.stopword_remover.remove(text)
    
    def stemming(self, text):
        """Stemming menggunakan Sastrawi"""
        return self.stemmer.stem(text)
    
    def preprocess(self, text):
        """Proses lengkap preprocessing"""
        # 1. Clean text
        text = self.clean_text(text)
        
        # 2. Case folding
        text = self.case_folding(text)
        
        # 3. Remove stopwords
        text = self.remove_stopwords(text)
        
        # 4. Stemming
        text = self.stemming(text)
        
        # Final cleaning: hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts):
        """Preprocess dalam batch"""
        return [self.preprocess(text) for text in texts]

def encode_labels(df):
    """Encode label sentimen menjadi numerik"""
    label_mapping = {
        'Positive': 1,
        'Neutral': 0,
        'Negative': -1
    }
    
    df['Sentiment_Encoded'] = df['Sentiment'].map(label_mapping)
    
    print("Label encoding:")
    print(f"Positive -> 1")
    print(f"Neutral -> 0")
    print(f"Negative -> -1")
    
    return df

def load_data():
    """Load dataset dengan auto-detect encoding"""
    try:
        # Coba beberapa encoding
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv('dataset_combined.csv', encoding=encoding)
                print(f"✅ Dataset loaded dengan encoding: {encoding}")
                print(f"Dataset shape: {df.shape}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        raise Exception("❌ Gagal membaca dataset dengan semua encoding yang dicoba")
    except FileNotFoundError:
        print("❌ File 'dataset_combined.csv' tidak ditemukan!")
        print("Jalankan 'data_understanding.py' terlebih dahulu.")
        return None

def main():
    """Main function"""
    print("="*60)
    print("🔄 DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\n📊 Data awal: {len(df)} baris")
    
    # Initialize preprocessor
    print("\n⚙️ Initializing preprocessor...")
    preprocessor = TextPreprocessor()
    
    # Preprocess text
    print("\n🔄 Memproses teks (ini mungkin memakan waktu beberapa menit)...")
    df['Text_Cleaned'] = preprocessor.preprocess_batch(df['Text'].astype(str))
    
    # Hapus baris dengan teks kosong setelah preprocessing
    df = df[df['Text_Cleaned'].str.strip() != '']
    print(f"📊 Data setelah preprocessing: {len(df)} baris")
    
    # Encode labels
    print("\n🏷️ Encoding labels...")
    df = encode_labels(df)
    
    # Simpan hasil preprocessing
    output_file = 'dataset_preprocessed.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Dataset preprocessed tersimpan sebagai '{output_file}' (UTF-8)")
    
    # Simpan preprocessor untuk digunakan nanti
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("✅ Preprocessor tersimpan sebagai 'preprocessor.pkl'")
    
    # Tampilkan sample hasil preprocessing
    print("\n" + "="*60)
    print("📝 SAMPLE HASIL PREPROCESSING")
    print("="*60)
    sample_df = df[['Text', 'Text_Cleaned', 'Sentiment']].head(5)
    for idx, row in sample_df.iterrows():
        print(f"\nOriginal: {row['Text'][:80]}...")
        print(f"Cleaned:  {row['Text_Cleaned']}")
        print(f"Label:    {row['Sentiment']}")
        print("-" * 60)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()

