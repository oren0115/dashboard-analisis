"""
📊 Dashboard Streamlit - Visualisasi Interaktif Sentimen
Dashboard untuk analisis sentimen dan prediksi real-time
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from preprocessing import TextPreprocessor
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4 0%, #6fa8dc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0 0 1.25rem 0;
    }
    .subtle { color: #6b7280; font-size: .95rem; text-align: center; margin-bottom: 1.5rem; }
    .card { background:#fff; border:1px solid #edf2f7; border-radius:12px; padding:16px 18px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
    .badge { display:inline-block; padding:.35rem .7rem; border-radius:999px; font-weight:600; font-size:.9rem; }
    .badge-pos { background:#e8f7ef; color:#139a43; }
    .badge-neg { background:#fde8e8; color:#d92332; }
    .badge-neu { background:#eef2f7; color:#475569; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load dataset dengan caching dan auto-detect encoding"""
    try:
        # Coba beberapa encoding
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv('dataset_combined.csv', encoding=encoding)
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return None
    except:
        return None

@st.cache_resource
def load_model():
    """Load model dengan caching"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model_data, preprocessor
    except:
        return None, None

def create_wordcloud(text, title):
    """Buat word cloud"""
    if len(text) == 0:
        return None
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig

def predict_sentiment(text, model_data, preprocessor):
    """Prediksi sentimen dari teks"""
    # Preprocess
    text_cleaned = preprocessor.preprocess(text)
    
    # Vectorize
    vectorizer = model_data['vectorizer']
    text_vectorized = vectorizer.transform([text_cleaned])
    
    # Predict
    model = model_data['model']
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Map prediction
    label_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    sentiment = label_mapping[prediction]
    
    # Map probabilities berdasarkan classes
    classes = model.classes_
    prob_mapped = [0.0, 0.0, 0.0]  # [Negative, Neutral, Positive]
    for i, cls in enumerate(classes):
        if cls == -1:
            prob_mapped[0] = probabilities[i]
        elif cls == 0:
            prob_mapped[1] = probabilities[i]
        elif cls == 1:
            prob_mapped[2] = probabilities[i]
    
    return sentiment, prob_mapped, text_cleaned

def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Analisis cepat dan sederhana untuk data sentimen</div>', unsafe_allow_html=True)
    px.defaults.template = "plotly_white"
    # Palet ramah color-blind (Blue/Orange/Green)
    px.defaults.color_discrete_sequence = ["#0072B2", "#E69F00", "#009E73"]
    
    # Sidebar
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Pilih Halaman",
        ["Beranda", "Analisis Data", "Prediksi Sentimen", "Evaluasi", "Visualisasi"]
    )
    
    # Load data
    with st.spinner('Memuat data...'):
        df = load_data()
    with st.spinner('Memuat model...'):
        model_data, preprocessor = load_model()
    
    if page == "Beranda":
        st.header("Selamat Datang di Sentiment Analysis Dashboard")
        st.markdown("""
        Dashboard ini menyediakan berbagai fitur untuk analisis sentimen:
        
        - Analisis Data: Eksplorasi dataset dan statistik
        - Prediksi Sentimen: Prediksi sentimen dari teks baru
        - Visualisasi: Grafik dan word cloud
        
        Gunakan menu di sidebar untuk navigasi.
        """)
        
        if df is not None:
            st.success(f"✅ Dataset dimuat: {len(df)} baris data")
        else:
            st.error("❌ Dataset tidak ditemukan. Pastikan file 'dataset_combined.csv' ada.")
        
        if model_data is not None:
            st.success("✅ Model ML dimuat dan siap digunakan")
        else:
            st.error("❌ Model tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
    
    elif page == "Analisis Data":
        st.header("Analisis Dataset")
        
        if df is None:
            st.error("Dataset tidak ditemukan!")
            return
        
        # Filter controls
        st.subheader("Filter Data")
        fcol1, fcol2, fcol3 = st.columns([2,2,2])
        with fcol1:
            date_min = pd.to_datetime(df['Date']).min().date()
            date_max = pd.to_datetime(df['Date']).max().date()
            date_range = st.date_input("Rentang Tanggal", (date_min, date_max))
        with fcol2:
            sentiments = st.multiselect("Sentimen", ['Positive','Neutral','Negative'], default=['Positive','Neutral','Negative'])
        with fcol3:
            keyword = st.text_input("Keyword (opsional)")

        # Apply filters
        df['_dt'] = pd.to_datetime(df['Date']).dt.date
        df_filtered = df[(df['_dt'] >= date_range[0]) & (df['_dt'] <= date_range[1])]
        if sentiments:
            df_filtered = df_filtered[df_filtered['Sentiment'].isin(sentiments)]
        if keyword:
            df_filtered = df_filtered[df_filtered['Text'].str.contains(keyword, case=False, na=False)]

        # Statistik umum
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", len(df_filtered))
        col2.metric("Positive", len(df_filtered[df_filtered['Sentiment'] == 'Positive']))
        col3.metric("Negative", len(df_filtered[df_filtered['Sentiment'] == 'Negative']))
        col4.metric("Neutral", len(df_filtered[df_filtered['Sentiment'] == 'Neutral']))
        
        # Distribusi sentimen
        st.subheader("Distribusi Sentimen")
        sentiment_counts = df_filtered['Sentiment'].value_counts()
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                    labels={'x': 'Sentimen', 'y': 'Jumlah'},
                    title='Distribusi Sentimen',
                    text=sentiment_counts.values)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Panjang teks
        st.subheader("Analisis Panjang Teks")
        fig = px.box(df_filtered, x='Sentiment', y='Length_Text',
                    title='Distribusi Panjang Teks per Sentimen',
                    color='Sentiment')
        st.plotly_chart(fig, use_container_width=True)
        
        # Tren waktu
        st.subheader("Tren Sentimen dari Waktu ke Waktu")
        daily_sentiment = df_filtered.groupby(['_dt', 'Sentiment']).size().reset_index(name='Count')
        
        fig = px.line(daily_sentiment, x='_dt', y='Count', 
                     color='Sentiment',
                     title='Tren Sentimen Harian',
                     labels={'_dt': 'Tanggal', 'Count': 'Jumlah Tweet'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel data
        st.subheader("Preview Data")
        limit = st.slider("Tampilkan N baris pertama", 10, 200, 100, 10)
        st.dataframe(df_filtered[['Date', 'Text', 'Sentiment', 'Length_Text']].head(limit))
    
    elif page == "Prediksi Sentimen":
        st.header("Prediksi Sentimen dari Teks")
        
        if model_data is None or preprocessor is None:
            st.error("Model tidak dimuat. Pastikan file model sudah ada.")
            return
        
        # Input teks
        st.subheader("Masukkan Teks untuk Prediksi")
        text_input = st.text_area("Teks:", height=100, 
                                 placeholder="Masukkan teks yang ingin dianalisis sentimennya...")
        
        if st.button("Prediksi Sentimen", type="primary"):
            if text_input.strip():
                # Predict
                sentiment, probabilities, text_cleaned = predict_sentiment(
                    text_input, model_data, preprocessor
                )
                
                # Tampilkan hasil - versi sederhana & rapi
                badge_class = {
                    'Positive': 'badge badge-pos',
                    'Negative': 'badge badge-neg',
                    'Neutral': 'badge badge-neu'
                }[sentiment]
                st.markdown(f"<div class='card'><div>Hasil Prediksi</div><div class='{badge_class}' style='margin-top:.35rem'>{sentiment}</div></div>", unsafe_allow_html=True)

                # Confidence bar chart
                prob_df = pd.DataFrame({
                    'Label': ['Negative','Neutral','Positive'],
                    'Prob': probabilities
                })
                fig = px.bar(prob_df, x='Label', y='Prob', range_y=[0,1], title='Confidence', text='Prob')
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                # Text cleaned
                with st.expander("Teks setelah Preprocessing"):
                    st.text(text_cleaned)
            else:
                st.warning("Masukkan teks terlebih dahulu!")
        
        # Batch prediction
        st.subheader("Prediksi Batch")
        batch_text = st.text_area("Masukkan beberapa teks (satu per baris):", 
                                  height=150,
                                  placeholder="Teks 1\nTeks 2\nTeks 3")
        
        if st.button("Prediksi Batch", type="secondary"):
            if batch_text.strip():
                texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                results = []
                with st.spinner('Memproses prediksi batch...'):
                    for text in texts:
                        sentiment, probabilities, text_cleaned = predict_sentiment(
                            text, model_data, preprocessor
                        )
                        results.append({
                            'Text': text,
                            'Sentiment': sentiment,
                            'Positive': probabilities[2],
                            'Neutral': probabilities[1],
                            'Negative': probabilities[0]
                        })
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button('Unduh Hasil CSV', data=csv, file_name='prediksi_batch.csv', mime='text/csv')
            else:
                st.warning("Masukkan teks terlebih dahulu!")

    elif page == "Evaluasi":
        st.header("Evaluasi Model (Quick Eval)")
        if df is None or not model_data or not preprocessor:
            st.info("Model/dataset belum siap. Jalankan training dan pastikan file tersedia.")
        else:
            n_samples = st.slider("Jumlah sampel untuk evaluasi", 100, 2000, 500, 100)
            sample_df = df.sample(min(n_samples, len(df)), random_state=42).copy()
            with st.spinner('Menghitung evaluasi...'):
                # Ground truth encoding
                map_label = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
                y_true = sample_df['Sentiment'].map(map_label).values
                # Predict
                texts_cleaned = preprocessor.preprocess_batch(sample_df['Text'].astype(str).tolist())
                vec = model_data['vectorizer']
                X_vec = vec.transform(texts_cleaned)
                mdl = model_data['model']
                y_pred = mdl.predict(X_vec)
            # Confusion matrix
            from sklearn.metrics import confusion_matrix, classification_report
            labels = [-1,0,1]
            disp_labels = ['Negative','Neutral','Positive']
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x='Predicted', y='True', color='Count'))
            fig_cm.update_xaxes(ticktext=disp_labels, tickvals=[0,1,2])
            fig_cm.update_yaxes(ticktext=disp_labels, tickvals=[0,1,2])
            st.plotly_chart(fig_cm, use_container_width=True)
            # Report ringkas
            report = classification_report(y_true, y_pred, labels=labels, target_names=disp_labels, output_dict=True, zero_division=0)
            st.write({k: {m: round(v,3) for m,v in report[k].items() if m in ['precision','recall','f1-score']} for k in disp_labels+['weighted avg']})
            # Contoh kesalahan
            sample_df['y_true'] = y_true
            sample_df['y_pred'] = y_pred
            inv_map = {-1:'Negative',0:'Neutral',1:'Positive'}
            wrong = sample_df[sample_df['y_true']!=sample_df['y_pred']][['Text','y_true','y_pred']].head(10).copy()
            if not wrong.empty:
                wrong['y_true'] = wrong['y_true'].map(inv_map)
                wrong['y_pred'] = wrong['y_pred'].map(inv_map)
                st.subheader('Contoh Prediksi Salah')
                st.dataframe(wrong, use_container_width=True)
    
    elif page == "Visualisasi":
        st.header("Visualisasi Data")
        
        if df is None:
            st.error("Dataset tidak ditemukan!")
            return
        
        # Word Cloud
        st.subheader("Word Cloud per Sentimen")
        use_clean = st.checkbox("Gunakan teks hasil preprocessing (butuh preprocessor)", value=False, disabled=(preprocessor is None))
        sentiment_type = st.selectbox("Pilih Sentimen:", ['Positive', 'Negative', 'Neutral', 'Semua'])
        
        def build_text(df_base):
            if use_clean and preprocessor is not None:
                texts = preprocessor.preprocess_batch(df_base['Text'].astype(str).tolist())
                return ' '.join(texts)
            return ' '.join(df_base['Text'].astype(str))
        
        if sentiment_type == 'Semua':
            text = build_text(df)
            title = 'Word Cloud - Semua Sentimen'
        else:
            text = build_text(df[df['Sentiment'] == sentiment_type])
            title = f'Word Cloud - {sentiment_type} Sentiment'
        
        if len(text) > 0:
            fig = create_wordcloud(text, title)
            if fig:
                st.pyplot(fig)
        else:
            st.warning("Tidak ada data untuk sentimen ini")
        
        # Top words
        st.subheader("Kata-Kata Paling Sering Muncul")
        
        from collections import Counter
        import re
        
        all_words = []
        for text in df['Text'].astype(str):
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(20)
        
        top_words_df = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
        fig = px.bar(top_words_df, x='Frekuensi', y='Kata', 
                    orientation='h',
                    title='Top 20 Kata Paling Sering Muncul')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

