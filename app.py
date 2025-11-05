"""
🚀 API Server - Flask API untuk Prediksi Sentimen
API endpoint untuk melakukan prediksi sentimen dari teks baru
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd
from preprocessing import TextPreprocessor
import os

app = Flask(__name__)

# Load model dan preprocessor
model_data = None
preprocessor = None

def load_model():
    """Load model dan preprocessor"""
    global model_data, preprocessor
    
    try:
        # Load model
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Load preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        print("✅ Model dan preprocessor berhasil dimuat")
        return True
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("Pastikan file 'best_model.pkl' dan 'preprocessor.pkl' sudah ada!")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Sentiment Analysis API',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Predict sentiment from text',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model_data is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment from text"""
    if model_data is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure model files exist'
        }), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Please provide "text" in JSON body'
            }), 400
        
        text = data['text']
        
        # Preprocess text
        text_cleaned = preprocessor.preprocess(text)
        
        # Vectorize
        vectorizer = model_data['vectorizer']
        text_vectorized = vectorizer.transform([text_cleaned])
        
        # Predict
        model = model_data['model']
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Map prediction to label
        label_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        sentiment = label_mapping[prediction]
        
        # Get probabilities - perlu mapping yang benar berdasarkan classes_
        classes = model.classes_
        prob_mapping = {}
        for i, cls in enumerate(classes):
            if cls == -1:
                prob_mapping['Negative'] = float(probabilities[i])
            elif cls == 0:
                prob_mapping['Neutral'] = float(probabilities[i])
            elif cls == 1:
                prob_mapping['Positive'] = float(probabilities[i])
        
        return jsonify({
            'text': text,
            'text_cleaned': text_cleaned,
            'sentiment': sentiment,
            'confidence': {
                'negative': prob_mapping['Negative'],
                'neutral': prob_mapping['Neutral'],
                'positive': prob_mapping['Positive']
            },
            'prediction': int(prediction)
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict sentiment from multiple texts"""
    if model_data is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure model files exist'
        }), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Please provide "texts" array in JSON body'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'Invalid request',
                'message': '"texts" must be an array'
            }), 400
        
        # Preprocess texts
        texts_cleaned = preprocessor.preprocess_batch(texts)
        
        # Vectorize
        vectorizer = model_data['vectorizer']
        texts_vectorized = vectorizer.transform(texts_cleaned)
        
        # Predict
        model = model_data['model']
        predictions = model.predict(texts_vectorized)
        probabilities = model.predict_proba(texts_vectorized)
        
        # Map predictions to labels
        label_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        classes = model.classes_
        
        results = []
        for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
            sentiment = label_mapping[pred]
            
            # Map probabilities berdasarkan classes
            prob_dict = {}
            for j, cls in enumerate(classes):
                if cls == -1:
                    prob_dict['negative'] = float(prob[j])
                elif cls == 0:
                    prob_dict['neutral'] = float(prob[j])
                elif cls == 1:
                    prob_dict['positive'] = float(prob[j])
            
            results.append({
                'text': text,
                'text_cleaned': texts_cleaned[i],
                'sentiment': sentiment,
                'confidence': prob_dict,
                'prediction': int(pred)
            })
        
        return jsonify({
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 MEMULAI SENTIMENT ANALYSIS API")
    print("="*60)
    
    # Load model
    if load_model():
        print("\n✅ Server siap!")
        print("📡 API akan berjalan di: http://localhost:5000")
        print("\nContoh penggunaan:")
        print("  POST http://localhost:5000/predict")
        print('  Body: {"text": "Saya sangat senang hari ini!"}')
        print("\n" + "="*60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Gagal memuat model. Pastikan file model sudah ada.")

