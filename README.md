# Batik Classification API - KarsaNusa

This project is a Flask-based application for classifying various types of Batik patterns from image inputs using a deep learning model.

## Features
- **Image Classification:** Classifies uploaded images into one of the Batik categories using a pre-trained TensorFlow model.
- **Detailed Metadata:** Provides detailed information about the classified Batik pattern, including its name, origin, description, and a reference image URL.
- **REST API:** Exposes endpoints for Batik classification and metadata retrieval.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KarsaNusa/karsanusa-machine-learning.git
   cd karsanusa-machine-learning
   ```
2. Create and activate a virtual environment:
    ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Ensure the following files are present in the project directory:
   - `batik_model.h5`: Pre-trained Batik classification model.

## Usage

1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Use the following API endpoints:

### Predict Batik Type
- **Endpoint:** `POST /predict`
- **Description:** Accepts an image file and returns the top three predicted Batik types along with their confidence scores.
- **Request Form-Data:**
  - `image`: Image file to classify.
- **Response:**
  ```json
  {
    "listPredictions": [
      {
        "name": "<batik_name>",
        "identifier": "<batik_identifier>",
        "confidence": <confidence_score>
      },
      ...
    ]
  }
  ```

### Get Batik Details
- **Endpoint:** `GET /details/<identifier>`
- **Description:** Retrieves detailed information about a specific Batik type by its identifier.
- **Response:**
  ```json
  {
    "detailResponse": {
      "name": "<batik_name>",
      "location": "<batik_origin>",
      "description": "<description>",
      "imageUrl": "<image_url>"
    }
  }
  ```

## Dependencies
- Flask
- TensorFlow
- Pillow
- NumPy

## Model and Dataset
- The classification model (`batik_model.h5`) is trained on a dataset of Batik categories.
- Each category includes detailed metadata stored in the application for enriching API responses.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or new features. Ensure your code passes existing tests and is well-documented.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- The Batik dataset and cultural metadata are sourced from various publicly available references to promote the preservation of Indonesia's heritage.

---

Enjoy exploring and classifying Indonesia's beautiful Batik patterns!
