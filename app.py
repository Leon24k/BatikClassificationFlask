from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Path ke model
MODEL_PATH = 'batik_model.h5'

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
    18: 'batik-sogan',
    19: 'batik-tambal'
}

details = {
    "batik-bali": {
        "name": "Batik Bali",
        "location": "Bali",
        "description": "Batik Bali adalah salah satu jenis batik yang berasal dari pulau Bali, Indonesia. Batik ini dikenal dengan motif yang sangat khas, menggabungkan unsur-unsur alam dan budaya Bali, seperti bunga, daun, serta berbagai elemen spiritual yang mencerminkan kekayaan budaya dan keindahan alam Bali. Batik Bali biasanya memiliki warna-warna cerah dan kontras, dengan pengaruh seni Bali yang kental. Motif-motif batik Bali sering kali menggambarkan kehidupan sehari-hari masyarakat Bali, serta mitos dan simbol-simbol tradisional yang sangat berhubungan dengan agama Hindu Bali.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-bali.jpg"
    },
    "batik-betawi": {
        "name": "Batik Betawi",
        "location": "Jakarta",
        "description": "Batik Betawi berasal dari Jakarta dan mengandung unsur-unsur budaya Melayu, Tionghoa, Arab, dan Eropa yang berbaur, mencerminkan keberagaman yang ada di ibu kota Indonesia. Batik ini memiliki ciri khas dengan motif yang beragam dan penuh warna, mulai dari pola geometris hingga motif flora yang menggambarkan keindahan alam, serta nuansa budaya Betawi yang kaya. Motif-motif Batik Betawi menggambarkan kebudayaan masyarakat Betawi, seperti rumah adat, kehidupan sehari-hari, serta cerita-cerita rakyat yang sudah turun temurun. Penggunaan warna cerah dan motif yang dinamis menjadikan Batik Betawi sangat mencolok dan menarik, sering dipakai pada acara adat dan budaya Jakarta.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-betawi.jpg"
    },
    "batik-celup": {
        "name": "Batik Celup",
        "location": "Solo",
        "description": "Batik Celup adalah teknik batik yang menggunakan metode celup untuk memberikan pewarnaan pada kain. Teknik ini memungkinkan terciptanya pola yang sangat khas dan unik, dengan warna-warna yang alami dan sering kali lembut. Batik Celup memberikan kesan yang elegan dengan perpaduan warna yang halus, serta memiliki kekhasan pada motif yang dihasilkan, yang sering kali mengambil inspirasi dari alam sekitar, seperti bunga dan daun. Batik ini sering menggunakan pewarna alami dan teknik pencelupan yang berulang untuk menghasilkan motif yang lebih kompleks dan warna yang lebih dalam, menjadikannya pilihan favorit dalam busana yang lebih formal dan bernuansa alam.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-celup.jpg"
    },
    "batik-cendrawasih": {
        "name": "Batik Cendrawasih",
        "location": "Papua",
        "description": "Batik Cendrawasih mengangkat motif burung cendrawasih yang dikenal dengan keindahan dan keanggunannya. Burung ini menjadi simbol keindahan alam, serta kebanggaan dan keagungan Indonesia. Motif batik ini memanfaatkan keindahan bentuk burung yang simetris, dengan desain yang elegan dan menawan. Dalam Batik Cendrawasih, burung ini sering kali digambarkan dalam posisi terbang, menggambarkan kebebasan dan kemegahan, menggunakan warna-warna cerah seperti merah, kuning, dan hijau yang menambah kesan dinamis dan hidup. Batik ini banyak digunakan pada pakaian formal maupun acara adat sebagai simbol kemuliaan dan keindahan.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-cendrawasih.jpg"
    },
    "batik-ceplok": {
        "name": "Batik Ceplok",
        "location": "Solo",
        "description": "Batik Ceplok memiliki pola yang sangat sederhana namun sangat khas, berupa pola geometris yang berulang dan simetris, biasanya berbentuk kotak atau lingkaran. Motif ini melambangkan ketenangan, keseimbangan, dan harmoni dalam hidup. Batik Ceplok sering kali mengandung simbolisme filosofis yang mendalam, seperti keharmonisan dalam kehidupan dan kesederhanaan. Pola yang berulang menciptakan kesan yang tertata rapi dan teratur, sangat cocok untuk pakaian sehari-hari atau acara adat yang mengedepankan nilai-nilai kedamaian dan kestabilan.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-ceplok.jpg"
    },
    "batik-ciamis": {
        "name": "Batik Ciamis",
        "location": "Ciamis, Jawa Barat",
        "description": "Batik Ciamis berasal dari daerah Ciamis di Jawa Barat dan dikenal dengan motif yang menggambarkan kehidupan sehari-hari serta alam sekitar. Batik ini menggunakan warna-warna alami yang lembut dan lebih gelap, yang memberi kesan tradisional dan elegan. Motif yang paling sering ditemukan adalah flora dan fauna setempat, yang mencerminkan kehidupan masyarakat pedesaan yang erat dengan alam. Batik Ciamis menggambarkan keharmonisan antara manusia dan alam, dengan pola yang sederhana namun penuh makna, serta sering digunakan dalam berbagai acara adat dan kebudayaan di Jawa Barat.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-ciamis.jpg"
    },
    "batik-garutan": {
        "name": "Batik Garutan",
        "location": "Garut, Jawa Barat",
        "description": "Batik Garutan berasal dari Garut, Jawa Barat, dan terkenal dengan keunikannya yang terinspirasi dari alam sekitar dan budaya lokal. Batik ini memiliki pola yang beragam, dari bunga hingga motif abstrak yang mencerminkan keindahan alam dan kehidupan sehari-hari masyarakat Garut. Batik Garutan biasanya menggunakan pewarnaan alami yang menciptakan warna-warna yang kaya dan mendalam, memberikan kesan artistik yang sangat kuat. Batik ini sering dipakai dalam berbagai acara adat dan kebudayaan, serta sebagai simbol identitas daerah yang kuat dan kaya akan tradisi.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-garutan.jpg"
    },
    "batik-gentongan": {
        "name": "Batik Gentongan",
        "location": "Cirebon",
        "description": "Batik Gentongan berasal dari Garut dan menggunakan teknik pewarnaan yang disebut gentong, yang memungkinkan terciptanya warna-warna yang kaya dan dalam. Motif-motif Batik Gentongan sangat dipengaruhi oleh alam, seperti daun, bunga, dan elemen-elemen alami lainnya yang mencerminkan kehidupan masyarakat Garut yang erat dengan alam. Teknik ini menciptakan pola-pola yang indah dan sangat detail, dengan warna-warna yang lebih natural, seperti merah, hitam, dan cokelat, menjadikannya pilihan favorit untuk pakaian adat dan formal.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-gentongan.jpg"
    },
    "batik-kawung": {
        "name": "Batik Kawung",
        "location": "Yogyakarta",
        "description": "Batik Kawung adalah salah satu motif batik yang paling terkenal dan ikonik di Indonesia. Motif ini berbentuk lingkaran atau oval yang saling terhubung dalam pola yang sangat simetris dan teratur, mencerminkan keseimbangan dan harmoni dalam kehidupan. Motif Kawung sering digunakan dalam pakaian resmi, terutama dalam upacara adat Jawa, dan dianggap memiliki makna filosofis yang mendalam, menggambarkan kesucian, kesabaran, dan ketenangan dalam hidup. Warna-warna yang digunakan dalam Batik Kawung biasanya lebih lembut, dengan kombinasi warna natural seperti cokelat, putih, dan biru.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-kawung.jpg"
    },
    "batik-keraton": {
        "name": "Batik Keraton",
        "location": "Yogyakarta",
        "description": "Batik Keraton adalah batik yang berasal dari keraton atau istana Jawa, dengan motif yang sangat elegan dan mewah. Motif Batik Keraton biasanya mencerminkan status sosial dan kekuasaan, dengan pola-pola yang rumit dan sangat terperinci, seperti bunga, burung, dan motif-motif alam lainnya. Batik ini sering kali digunakan oleh kalangan kerajaan atau bangsawan dalam upacara resmi, serta menggambarkan kemewahan dan kehormatan. Motif yang digunakan sangat halus dan memiliki nilai seni yang tinggi, dengan penggunaan warna emas, merah, dan biru yang mewah.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-keraton.jpg"
    },
    "batik-lasem": {
        "name": "Batik Lasem",
        "location": "Lasem, Jawa Tengah",
        "description": "Batik Lasem berasal dari Lasem, Jawa Tengah, yang dikenal dengan pengaruh budaya Tionghoa yang kental. Batik ini memiliki pola-pola yang beragam, menggabungkan elemen-elemen Jawa dan Tionghoa, dengan penggunaan warna merah yang dominan, menciptakan kesan yang kuat dan mencolok. Batik Lasem menggambarkan keberagaman budaya yang hidup di Lasem, serta simbolisme yang mencerminkan keberuntungan, kemakmuran, dan kebahagiaan. Batik ini sangat populer dalam pakaian adat dan digunakan dalam berbagai perayaan serta acara tradisional.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-lasem.jpg"
    },
    "batik-megamendung": {
        "name": "Batik Megamendung",
        "location": "Cirebon",
        "description": "Batik Megamendung berasal dari Cirebon, yang terkenal dengan motif awan yang sangat khas dan melambangkan kedamaian serta ketenangan. Batik ini menggambarkan awan yang bergerak di langit, dengan warna-warna cerah seperti biru, merah, kuning, dan hijau yang menciptakan kesan dinamis dan penuh semangat. Motif Batik Megamendung sangat erat dengan budaya Cirebon, yang menganggap awan sebagai simbol perlindungan dan kesejahteraan. Batik ini banyak digunakan dalam pakaian sehari-hari maupun dalam acara adat dan keagamaan.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-megamendung.jpg"
    },
    "batik-parang": {
        "name": "Batik Parang",
        "location": "Yogyakarta",
        "description": "Batik Parang adalah salah satu motif batik yang paling dikenal di Indonesia, dengan pola garis diagonal yang saling berputar, menggambarkan keteguhan dan kekuatan. Motif ini memiliki simbolisme yang dalam dalam budaya Jawa, yang melambangkan perjuangan hidup dan semangat juang. Batik Parang sering digunakan oleh kalangan kerajaan sebagai simbol status dan kehormatan, dengan pola yang elegan dan penuh makna, serta menggunakan warna-warna yang lebih gelap dan menonjolkan kesan keanggunan serta kemuliaan.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-parang.jpg"
    },
    "batik-pekalongan": {
        "name": "Batik Pekalongan",
        "location": "Pekalongan",
        "description": "Batik Pekalongan berasal dari kota Pekalongan di Jawa Tengah dan sangat terkenal dengan keragaman motifnya yang menggambarkan kebudayaan yang sangat beragam, dari Tionghoa, Arab, hingga Eropa. Batik ini memiliki ciri khas dengan pola yang sangat variatif dan penggunaan warna-warna cerah yang mencerminkan kehidupan yang ceria dan dinamis. Batik Pekalongan sering digunakan dalam pakaian sehari-hari, serta dalam acara perayaan dan acara adat di Jawa Tengah, dan menjadi simbol keberagaman budaya yang hidup di Pekalongan.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-pekalongan.jpg"
    },
    "batik-priangan": {
        "name": "Batik Priangan",
        "location": "Bandung",
        "description": "Batik Priangan berasal dari daerah Priangan di Jawa Barat dan dikenal dengan pola-pola yang menggambarkan keindahan alam dan kehidupan pedesaan. Batik ini menggunakan warna-warna alami yang lebih lembut, seperti hijau, cokelat, dan kuning, serta motif yang mencerminkan kedamaian dan ketenangan alam. Batik Priangan banyak digunakan dalam acara adat dan budaya Jawa Barat, serta menjadi simbol kedamaian dan keharmonisan antara manusia dan alam.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-priangan.jpg"
    },
    "batik-sekar": {
        "name": "Batik Sekar",
        "location": "Yogyakarta",
        "description": "Batik Sekar memiliki motif bunga atau tanaman yang dominan, dengan pola yang indah dan penuh detail. Batik ini sering kali digunakan pada pakaian wanita dan menggambarkan keanggunan serta kesuburan alam. Motif bunga dalam Batik Sekar mencerminkan kecantikan dan kelembutan, serta sering digunakan dalam acara adat yang melibatkan wanita, seperti pernikahan dan upacara keagamaan, sebagai simbol keberkahan dan kehidupan baru.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-sekar.jpg"
    },
    "batik-sidoluhur": {
        "name": "Batik Sidoluhur",
        "location": "Magelang",
        "description": "Batik Sidoluhur berasal dari Sidoluhur, Jawa Tengah, dan dikenal dengan motif yang sangat sederhana namun sangat elegan. Batik ini sering menggambarkan kehidupan sehari-hari masyarakat Sidoluhur, dengan warna yang lebih gelap dan motif yang lebih natural. Batik Sidoluhur memiliki kesan tenang dan damai, serta sering digunakan dalam acara adat yang lebih formal sebagai simbol kedamaian dan ketenangan.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-sidoluhur.jpg"
    },
    "batik-sidomukti": {
        "name": "Batik Sidomukti",
        "location": "Cirebon",
        "description": "Batik Sidomukti memiliki motif yang kaya dengan filosofi kehidupan, menggambarkan perjuangan dan harapan dalam hidup. Motif ini sangat erat dengan budaya Jawa, yang melambangkan perjalanan hidup manusia yang penuh dengan rintangan dan tantangan. Batik Sidomukti sering kali digunakan dalam acara keagamaan atau upacara adat sebagai simbol harapan dan kekuatan dalam menghadapi kehidupan.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-sidomukti.jpg"
    },
    "batik-sogan": {
        "name": "Batik Sogan",
        "location": "Surakarta",
        "description": "Batik Sogan menggunakan warna dasar cokelat tua dengan motif yang lebih sederhana namun tetap memiliki nilai seni yang tinggi. Motif-motif dalam Batik Sogan sering kali menggambarkan ketenangan, kebijaksanaan, dan kedamaian. Batik ini banyak digunakan dalam pakaian formal dan acara adat, serta menjadi simbol kedamaian dan kesejahteraan dalam budaya Jawa.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-sogan.jpg"
    },
    "batik-tambal": {
        "name": "Batik Tambal",
        "location": "Jawa Tengah",
        "description": "Batik Tambal adalah jenis batik yang memiliki ciri khas dengan pola-pola tambalan atau potongan yang saling disusun untuk membentuk motif tertentu. Teknik ini menciptakan kesan dinamis dan berlapis, di mana berbagai potongan kain dengan motif berbeda disatukan untuk menciptakan desain yang unik dan berwarna-warni. Batik Tambal biasanya menggunakan warna-warna cerah dan kontras, mencerminkan kekayaan budaya lokal. Motif yang dihasilkan dapat bervariasi, tetapi sering kali menggambarkan elemen-elemen alam dan kehidupan sehari-hari masyarakat, seperti bunga, daun, serta pola geometris. Batik Tambal menjadi simbol kreativitas dan keragaman dalam seni batik, serta sering digunakan pada pakaian adat dan acara budaya di Jawa Tengah.",
        "imageUrl": "https://storage.googleapis.com/karsanusa-capstone.appspot.com/imageCDN/batik-tambal.jpg"
    }
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
        list_predictions = [
            {
                "name": labels[int(i)].replace("-", " ").title(),  # Nama rapi
                "identifier": labels[int(i)],  # Nama teknis
                "confidence": float(probabilities[i])  # Nilai confidence
            }
            for i in top_indices if int(i) in labels
        ]

        # Bungkus respons dalam key "listPredictions"
        return jsonify({
            "listPredictions": list_predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/details/<identifier>', methods=['GET'])
def get_details(identifier):
    # Cari apakah identifier ada dalam detail
    if identifier not in details:
        return jsonify({"error": "Detail tidak ditemukan"}), 404

    # Kembalikan detail terkait identifier
    return jsonify({
        "detailResponse": details[identifier]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)