import os
import io
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. GENEL KURULUM ---

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# Flask uygulamasını ve CORS'u yapılandır
app = Flask(__name__)
CORS(app)

# --- 2. MODEL VE VERİTABANI KURULUMU ---

# --- ⭐ YENİ ⭐: Gemini API Kurulumu
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini API başarıyla yapılandırıldı.")
except Exception as e:
    print(f"❌ Gemini API yapılandırılamadı: {e}")
    gemini_model = None

# --- Yerel Keras Modeli Kurulumu
try:
    MODEL_PATH = "food_model_mobilenet.h5"
    local_model = load_model(MODEL_PATH)
    print(f"✅ Yerel model '{MODEL_PATH}' başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Yerel model yüklenirken hata oluştu: {e}")
    local_model = None

# --- Besin Değeri Veritabanı Kurulumu
try:
    CSV_PATH = 'food101_with_nutrition_CLEAN.csv'
    df = pd.read_csv(CSV_PATH)
    df.dropna(subset=['calories'], inplace=True)
    df.drop_duplicates(subset=['class_name'], keep='first', inplace=True)
    nutrition_db = df.set_index('class_name')
    unique_classes = sorted(df['class_name'].unique())
    idx_to_class = {i: name for i, name in enumerate(unique_classes)}
    print("✅ Besin Değeri Veritabanı başarıyla hazırlandı.")
except Exception as e:
    print(f"❌ Besin Değeri Veritabanı hazırlanırken hata oluştu: {e}")
    nutrition_db = None


# --- 3. YARDIMCI FONKSİYONLAR ---

def prepare_image_for_local_model(img):
    """Gelen resmi yerel Keras modelinin beklediği formata dönüştürür."""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)


# --- ⭐ YENİ ⭐: Gemini ile Tahmin Fonksiyonu
def predict_with_gemini(image_path):
    """Verilen resim yolunu kullanarak Gemini API'den yemek adını tahmin eder."""
    if not gemini_model:
        print("❌ Gemini modeli yüklü değil, tahmin yapılamıyor.")
        return None

    print("[INFO] Gemini'ye sorgu gönderiliyor...")
    try:
        img = Image.open(image_path)
        prompt = """
            Bu resimdeki yiyeceği analiz et. 
            1. Yiyeceğin adını belirle.
            2. Bu yiyeceğin standart 100 gramlık porsiyonu için tahmini besin değerlerini hesapla.
            Cevabını, başka hiçbir açıklama veya metin eklemeden, SADECE aşağıdaki gibi bir JSON formatında ver:
            {
              "food_name": "Yemeğin Adı",
              "calories": <kalori_değeri_tamsayı>,
              "protein": <protein_gramı_ondalıklı>,
              "fat": <yağ_gramı_ondalıklı>
            }
            """
        response = gemini_model.generate_content([prompt, img])
        gemini_result = response.text.strip().replace("*", "")
        print(f"[INFO] Gemini Sonucu: {gemini_result}")
        return gemini_result

    except Exception as e:
        print(f"❌ Gemini API hatası: {e}")
        traceback.print_exc()
        return None


# --- ⭐ YENİ ⭐: Besin Değerlerini Alma Fonksiyonu
def get_nutrition_info(food_class_name):
    """Verilen yemek adı için veritabanından besin değerlerini çeker."""
    if nutrition_db is None:
        return 0.0, 0.0, 0.0  # Kalori, protein, yağ

    try:
        food_info = nutrition_db.loc[food_class_name]
        calories = float(food_info.get('calories', 0.0))
        protein = float(food_info.get('protein', 0.0))
        fats = float(food_info.get('fats', 0.0))
        return calories, protein, fats
    except (KeyError, TypeError):
        # Eğer yemek veritabanında bulunamazsa
        print(f"[WARN] Besin değeri '{food_class_name}' için veritabanında bulunamadı.")
        return 0.0, 0.0, 0.0


# --- 4. API ENDPOINT'İ ---
@app.route("/predict", methods=["POST"])
def predict():
    # --- Model ve dosya kontrolleri
    if local_model is None:
        return jsonify({'error': 'Yerel model yüklenemedi. Sunucu başlatılamıyor.'}), 500
    if 'image' not in request.files:
        return jsonify({"error": "Lütfen 'image' anahtarıyla bir resim dosyası gönderin."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'}), 400

    try:
        # --- Resmi kaydet ve yerel model için hazırla
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', 'last_received_food_image.jpg')
        file.save(file_path)

        pil_img = Image.open(file_path).convert("RGB")
        prepared_image = prepare_image_for_local_model(pil_img)

        # --- Yerel modelle tahmin yap
        predictions = local_model.predict(prepared_image)
        class_probabilities = predictions[0][0]
        local_predicted_calorie = predictions[1][0][0]  # Yerel modelin kalori tahmini

        predicted_index = np.argmax(class_probabilities)
        confidence = float(class_probabilities[predicted_index])
        class_name = idx_to_class[predicted_index]

        print(f"[INFO] Yerel Model Tahmini: {class_name}, Güven: {confidence:.2f}")

        # --- ⭐ GÜNCELLENDİ ⭐: Güven skoruna göre karar verme mantığı
        CONFIDENCE_THRESHOLD = 0.60  # Bu eşiği ihtiyacınıza göre ayarlayabilirsiniz

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"[INFO] Güven skoru ({confidence:.2f}) düşük. Gemini'ye danışılıyor...")
            gemini_prediction_name = predict_with_gemini(file_path)

            if gemini_prediction_name:
                # Gemini başarılı bir tahmin yaptı
                source = "Gemini API"
                final_food_name_readable = gemini_prediction_name
                # Gemini sonucunu veritabanı formatına çevir (örn: "Caesar Salad" -> "caesar_salad")
                food_class_name_for_db = gemini_prediction_name.lower().replace(' ', '_')
                # Besin değerlerini veritabanından bu yeni isme göre çek
                calories, protein, fat = get_nutrition_info(food_class_name_for_db)
                final_confidence_str = "N/A (Gemini)"
            else:
                # Gemini başarısız oldu, düşük güvenli yerel modele geri dön
                source = "Yerel Model (Düşük Güven)"
                final_food_name_readable = class_name.replace('_', ' ').title()
                calories, protein, fat = get_nutrition_info(class_name)
                # Düşük de olsa yerel modelin kalori tahminini kullanabiliriz
                if calories == 0.0: calories = local_predicted_calorie
                final_confidence_str = f"{confidence * 100:.2f}%"
        else:
            # Yerel modelin güveni yeterince yüksek
            source = "Yerel Model"
            final_food_name_readable = class_name.replace('_', ' ').title()
            # Yüksek güvende, hem veritabanından hem de modelden gelen kalori bilgisini kullanabiliriz.
            # Öncelik veritabanında olsun.
            db_calories, protein, fat = get_nutrition_info(class_name)
            calories = db_calories if db_calories > 0 else local_predicted_calorie
            final_confidence_str = f"{confidence * 100:.2f}%"

        # --- Sonucu JSON olarak hazırla
        result = {
            "food_name": final_food_name_readable,
            "source": source,
            "confidence": final_confidence_str,
            "nutritions": {
                "calories": int(round(calories, 0)),
                "protein_g": round(protein, 1),
                "fat_g": round(fat, 1)
            }
        }
        print(f"[SUCCESS] Sonuç gönderiliyor: {result}")
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Tahmin sırasında beklenmedik bir hata oluştu.", "details": str(e)}), 500


if __name__ == '__main__':
    # 'debug=True' geliştirme sırasında kullanışlıdır, ancak canlı (production) ortamda 'False' yapın.
    app.run(host='0.0.0.0', port=5000, debug=True)