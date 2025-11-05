"""
🤖 Model Training - Training Model Machine Learning
Script untuk training berbagai model klasifikasi sentimen
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

def load_data():
    """Load dataset yang sudah di-preprocess dengan auto-detect encoding"""
    try:
        # Coba beberapa encoding
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv('dataset_preprocessed.csv', encoding=encoding)
                print(f"✅ Dataset loaded dengan encoding: {encoding}")
                print(f"Dataset shape: {df.shape}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        raise Exception("❌ Gagal membaca dataset dengan semua encoding yang dicoba")
    except FileNotFoundError:
        print("❌ File 'dataset_preprocessed.csv' tidak ditemukan!")
        print("Jalankan 'preprocessing.py' terlebih dahulu.")
        return None

def prepare_features(df):
    """Siapkan features dan labels"""
    X = df['Text_Cleaned'].values
    y = df['Sentiment_Encoded'].values
    
    print(f"\n📊 Data preparation:")
    print(f"Features (X): {X.shape}")
    print(f"Labels (y): {y.shape}")
    print(f"Label distribution: {np.bincount(y + 1)}")  # +1 karena label -1,0,1
    
    return X, y

def extract_features(X_train, X_test, max_features=5000):
    """Ekstraksi fitur menggunakan TF-IDF"""
    print("\n🔤 Extracting features dengan TF-IDF...")
    print(f"Max features: {max_features}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigram dan bigram
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✅ Feature extraction selesai")
    print(f"Train features shape: {X_train_tfidf.shape}")
    print(f"Test features shape: {X_test_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("\n📊 Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    print("✅ Logistic Regression training selesai")
    return model

def train_svm(X_train, y_train):
    """Train SVM model"""
    print("\n📊 Training SVM...")
    model = SVC(
        kernel='linear',
        C=1.0,
        random_state=42,
        probability=True
    )
    model.fit(X_train, y_train)
    print("✅ SVM training selesai")
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\n📊 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest training selesai")
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluasi model dan return metrics"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"📊 EVALUASI MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Classification report - perlu mapping label yang benar
    # Get unique labels dan sort
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    label_names = []
    for label in unique_labels:
        if label == -1:
            label_names.append('Negative')
        elif label == 0:
            label_names.append('Neutral')
        elif label == 1:
            label_names.append('Positive')
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=unique_labels, 
                              target_names=label_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    plot_confusion_matrix(cm, model_name, label_names, unique_labels)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred
    }

def plot_confusion_matrix(cm, model_name, label_names, labels):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    filename = f"output/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix tersimpan di: {filename}")
    plt.close()

def compare_models(results):
    """Bandingkan performa semua model"""
    print("\n" + "="*60)
    print("📊 PERBANDINGAN MODEL")
    print("="*60)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score']]
    
    print(comparison_df.to_string(index=False))
    
    # Visualisasi perbandingan
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.2
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width, comparison_df[metric], width, 
               label=metric.replace('_', ' ').title(), color=color, alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Perbandingan Performa Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['model_name'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Visualisasi perbandingan tersimpan di: output/model_comparison.png")
    plt.close()

def save_model(model, vectorizer, model_name, filename):
    """Simpan model dan vectorizer"""
    model_data = {
        'model': model,
        'vectorizer': vectorizer
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ Model '{model_name}' tersimpan sebagai '{filename}'")

def main():
    """Main function"""
    print("="*60)
    print("🤖 MODEL TRAINING - KLASIFIKASI SENTIMEN")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Extract features
    vectorizer, X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)
    
    # Train models
    models = {}
    models['Logistic Regression'] = train_logistic_regression(X_train_tfidf, y_train)
    models['SVM'] = train_svm(X_train_tfidf, y_train)
    models['Random Forest'] = train_random_forest(X_train_tfidf, y_train)
    
    # Evaluate models
    results = []
    for model_name, model in models.items():
        result = evaluate_model(model, X_test_tfidf, y_test, model_name)
        results.append(result)
    
    # Compare models
    compare_models(results)
    
    # Pilih model terbaik berdasarkan F1-score
    best_model_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
    best_model_name = results[best_model_idx]['model_name']
    best_model = models[best_model_name]
    
    print(f"\n🏆 Model terbaik: {best_model_name}")
    print(f"   F1-Score: {results[best_model_idx]['f1_score']:.4f}")
    
    # Simpan model terbaik
    save_model(best_model, vectorizer, best_model_name, 'best_model.pkl')
    
    # Simpan semua model
    for model_name, model in models.items():
        filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        os.makedirs('models', exist_ok=True)
        save_model(model, vectorizer, model_name, filename)
    
    print("\n" + "="*60)
    print("✅ TRAINING SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()

