from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

app = Flask(__name__)

# Cargar el modelo y el tokenizador globalmente
print("Cargando modelo y tokenizador...")
model = TFBertForSequenceClassification.from_pretrained('bert_modelo_entrenado_V1')
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