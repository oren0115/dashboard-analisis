"""
📊 Data Understanding - Analisis Sentimen
Script untuk melakukan eksplorasi dan analisis awal terhadap dataset sentimen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import os

warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data():
    """Load dataset dari file CSV dengan auto-detect encoding"""
    print("📂 Loading dataset...")
    
    # Coba beberapa encoding yang umum digunakan
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    
    df_sentiment = None
    df_train = None
    
    # Load Sentiment1.csv
    for encoding in encodings:
        try:
            df_sentiment = pd.read_csv('Sentiment1.csv', encoding=encoding)
            print(f"✅ Sentiment1.csv loaded dengan encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"⚠️ Error dengan encoding {encoding}: {e}")
            continue
    
    if df_sentiment is None:
        raise Exception("❌ Gagal membaca Sentiment1.csv dengan semua encoding yang dicoba")
    
    # Load Train1.csv
    for encoding in encodings:
        try:
            df_train = pd.read_csv('Train1.csv', encoding=encoding)
            print(f"✅ Train1.csv loaded dengan encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"⚠️ Error dengan encoding {encoding}: {e}")
            continue
    
    if df_train is None:
        raise Exception("❌ Gagal membaca Train1.csv dengan semua encoding yang dicoba")
    
    print(f"Dataset Sentiment1.csv: Shape {df_sentiment.shape}")
    print(f"Dataset Train1.csv: Shape {df_train.shape}")
    
    return df_sentiment, df_train

def combine_datasets(df_sentiment, df_train):
    """Gabungkan dataset dan hapus duplikat"""
    print("\n🔄 Menggabungkan dataset...")
    df_combined = pd.concat([df_sentiment, df_train], ignore_index=True)
    print(f"Total data setelah digabungkan: {df_combined.shape}")
    print(f"Duplikat: {df_combined.duplicated().sum()}")
    
    # Hapus duplikat
    df_combined = df_combined.drop_duplicates()
    print(f"Total data setelah hapus duplikat: {df_combined.shape}")
    
    return df_combined

def analyze_data(df):
    """Analisis dasar dataset"""
    print("\n" + "="*50)
    print("📊 INFO DATASET")
    print("="*50)
    df.info()
    
    print("\n" + "="*50)
    print("📋 PREVIEW DATA")
    print("="*50)
    print(df.head())
    
    print("\n" + "="*50)
    print("❌ MISSING VALUES")
    print("="*50)
    missing = df.isnull().sum()
    print(missing)
    print(f"\nTotal missing: {missing.sum()}")

def analyze_sentiment_distribution(df):
    """Analisis distribusi sentimen"""
    print("\n" + "="*50)
    print("📈 DISTRIBUSI SENTIMEN")
    print("="*50)
    sentiment_counts = df['Sentiment'].value_counts()
    print(sentiment_counts)
    print(f"\nPersentase:")
    print(sentiment_counts / len(df) * 100)
    
    return sentiment_counts

def visualize_sentiment_distribution(sentiment_counts):
    """Visualisasi distribusi sentimen"""
    print("\n📊 Membuat visualisasi distribusi sentimen...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar Chart
    axes[0].bar(sentiment_counts.index, sentiment_counts.values, 
                color=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[0].set_title('Distribusi Sentimen (Bar Chart)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sentimen')
    axes[0].set_ylabel('Jumlah')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie Chart
    axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c', '#95a5a6'], startangle=90)
    axes[1].set_title('Distribusi Sentimen (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Visualisasi tersimpan di: output/sentiment_distribution.png")
    plt.close()

def analyze_text_length(df):
    """Analisis panjang teks"""
    print("\n" + "="*50)
    print("📏 STATISTIK PANJANG TEKS")
    print("="*50)
    print(df['Length_Text'].describe())
    
    # Visualisasi
    print("\n📊 Membuat visualisasi panjang teks...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['Length_Text'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribusi Panjang Teks', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Panjang Teks (karakter)')
    axes[0].set_ylabel('Frekuensi')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box Plot per Sentimen
    df.boxplot(column='Length_Text', by='Sentiment', ax=axes[1])
    axes[1].set_title('Panjang Teks per Sentimen', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sentimen')
    axes[1].set_ylabel('Panjang Teks (karakter)')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('')
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/text_length_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualisasi tersimpan di: output/text_length_analysis.png")
    plt.close()

def analyze_temporal_trends(df):
    """Analisis tren waktu"""
    print("\n📅 Menganalisis tren waktu...")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_only'] = df['Date'].dt.date
    
    # Sentimen per hari
    daily_sentiment = df.groupby(['Date_only', 'Sentiment']).size().unstack(fill_value=0)
    
    # Visualisasi tren
    plt.figure(figsize=(14, 6))
    daily_sentiment.plot(kind='line', marker='o', linewidth=2, markersize=4)
    plt.title('Tren Sentimen dari Waktu ke Waktu', fontsize=14, fontweight='bold')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Tweet')
    plt.legend(title='Sentimen')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/temporal_trends.png', dpi=300, bbox_inches='tight')
    print("✅ Visualisasi tersimpan di: output/temporal_trends.png")
    plt.close()

def create_wordclouds(df):
    """Membuat word cloud untuk setiap sentimen"""
    print("\n☁️ Membuat word cloud...")
    
    def create_wordcloud(text, title):
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             max_words=100).generate(text)
        return wordcloud
    
    # Gabungkan teks per sentimen
    positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Text'].astype(str))
    negative_text = ' '.join(df[df['Sentiment'] == 'Negative']['Text'].astype(str))
    neutral_text = ' '.join(df[df['Sentiment'] == 'Neutral']['Text'].astype(str))
    
    # Visualisasi Word Cloud
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    wordclouds = [
        create_wordcloud(positive_text, 'Positive Sentiment'),
        create_wordcloud(negative_text, 'Negative Sentiment'),
        create_wordcloud(neutral_text, 'Neutral Sentiment')
    ]
    
    titles = ['Positive Sentiment', 'Negative Sentiment', 'Neutral Sentiment']
    
    for ax, wc, title in zip(axes, wordclouds, titles):
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/wordclouds.png', dpi=300, bbox_inches='tight')
    print("✅ Visualisasi tersimpan di: output/wordclouds.png")
    plt.close()

def show_samples(df):
    """Tampilkan sample data per sentimen"""
    print("\n" + "="*50)
    print("📝 SAMPLE DATA")
    print("="*50)
    
    print("\n=== SAMPLE DATA POSITIVE ===")
    positive_samples = df[df['Sentiment'] == 'Positive']['Text'].head(3).values
    for i, sample in enumerate(positive_samples, 1):
        print(f"{i}. {sample[:100]}...")
    
    print("\n=== SAMPLE DATA NEGATIVE ===")
    negative_samples = df[df['Sentiment'] == 'Negative']['Text'].head(3).values
    for i, sample in enumerate(negative_samples, 1):
        print(f"{i}. {sample[:100]}...")
    
    print("\n=== SAMPLE DATA NEUTRAL ===")
    neutral_samples = df[df['Sentiment'] == 'Neutral']['Text'].head(3).values
    for i, sample in enumerate(neutral_samples, 1):
        print(f"{i}. {sample[:100]}...")

def save_combined_dataset(df):
    """Simpan dataset yang sudah digabungkan dengan UTF-8 encoding"""
    output_file = 'dataset_combined.csv'
    # Simpan dengan UTF-8 untuk konsistensi
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Dataset berhasil disimpan sebagai '{output_file}' (UTF-8)")
    return output_file

def main():
    """Main function"""
    print("="*60)
    print("📊 DATA UNDERSTANDING - ANALISIS SENTIMEN")
    print("="*60)
    
    # Load data
    df_sentiment, df_train = load_data()
    
    # Combine datasets
    df_combined = combine_datasets(df_sentiment, df_train)
    
    # Analyze data
    analyze_data(df_combined)
    
    # Analyze sentiment distribution
    sentiment_counts = analyze_sentiment_distribution(df_combined)
    visualize_sentiment_distribution(sentiment_counts)
    
    # Analyze text length
    analyze_text_length(df_combined)
    
    # Analyze temporal trends
    analyze_temporal_trends(df_combined)
    
    # Create word clouds
    create_wordclouds(df_combined)
    
    # Show samples
    show_samples(df_combined)
    
    # Save combined dataset
    save_combined_dataset(df_combined)
    
    print("\n" + "="*60)
    print("✅ DATA UNDERSTANDING SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()

