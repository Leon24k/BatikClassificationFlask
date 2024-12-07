from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Path ke model
MODEL_PATH = 'batik_classification_model.h5'

# Load model
model = load_model(MODEL_PATH)

# Label klasifikasi
labels = {
    0: 'batik-bali',
    1: 'batik-betawi',
    2: 'batik-celup',
    3: 'batik-cendrawasih',
    4: 'batik-ceplok',
    5: 'batik-ciamis',
    6: 'batik-garutan',
    7: 'batik-gentongan',
    8: 'batik-kawung',
    9: 'batik-keraton',
    10: 'batik-lasem',
    11: 'batik-megamendung',
    12: 'batik-parang',
    13: 'batik-pekalongan',
    14: 'batik-priangan',
    15: 'batik-sekar',
    16: 'batik-sidoluhur',
    17: 'batik-sidomukti',
    18: 'batik-sogan'
}

def preprocess_image(file, target_size=(224, 224)):
    """
    Preprocess image untuk dimasukkan ke model.
    file: FileStorage dari Flask request.
    target_size: Ukuran target untuk resize gambar.
    """
    img = Image.open(BytesIO(file.read())).convert('RGB')  # Pastikan mode RGB
    img = img.resize(target_size)  # Resize gambar
    img_array = img_to_array(img)  # Konversi ke array
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    img_array = img_array / 255.0  # Normalisasi
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah"}), 400

    file = request.files['image']

    try:
        # Preprocess gambar
        img_array = preprocess_image(file)

        # Lakukan prediksi
        predictions = model.predict(img_array)
        probabilities = predictions[0]

        # Urutkan hasil berdasarkan probabilitas tertinggi
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = {labels[i]: float(probabilities[i]) for i in top_indices}

        return jsonify({
            "predictions": top_predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
