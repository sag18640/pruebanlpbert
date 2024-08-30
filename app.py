from flask import Flask, request, jsonify
import os
import requests
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

app = Flask(__name__)

# URL de los archivos del modelo y el tokenizador
MODEL_URL = 'https://storage.googleapis.com/terceron/pruebas/tf_model.h5'
CONFIG_JSON = 'https://storage.googleapis.com/terceron/pruebas/config.json'
# TOKENIZER_URL = 'https://your-storage-service.com/path/to/bert_tokenizer_V1'
MODEL_PATH = '/tmp/bert_modelo_entrenado_V1'
TOKENIZER_PATH = '/tmp/bert_tokenizer_V1'

def download_file(url, path):
    """Descarga un archivo desde una URL si no existe en el sistema de archivos local."""
    if not os.path.exists(path):
        print(f'Descargando {path} desde {url}...')
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'{path} descargado exitosamente.')
    else:
        print(f'{path} encontrado localmente, usando la versión en caché.')

# Descargar el modelo y el tokenizador si no están presentes
download_file(MODEL_URL, MODEL_PATH)
# download_file(TOKENIZER_URL, TOKENIZER_PATH)
download_file(CONFIG_JSON, MODEL_PATH)

# Cargar el modelo y el tokenizador globalmente
print("Cargando modelo y tokenizador...")
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained('bert_tokenizer_V1')
print("Modelo y tokenizador cargados exitosamente.")

def predict_search_type(search_term):
    inputs = tokenizer(search_term, truncation=True, padding=True, max_length=128, return_tensors="tf")
    outputs = model(inputs)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
    confidence = probabilities[0][predicted_class].numpy().item()
    
    class_names = ['Marca', 'Producto', 'Categoría']
    predicted_label = class_names[predicted_class]
    
    return predicted_label, confidence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    search_term = data['search_term']
    
    predicted_label, confidence = predict_search_type(search_term)
    
    return jsonify({
        'search_term': search_term,
        'prediction': predicted_label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=False)