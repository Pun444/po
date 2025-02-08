import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# โหลดโมเดลที่เทรนไว้
model = load_model('C:/Dataset/Lung/gunnet.h5')

# รายชื่อโรคและข้อมูลคำแนะนำ
disease_info = {
    "bron": {"description": "หลอดลมอักเสบ ทำให้ไอเรื้อรังและหายใจลำบาก", "care": "ดื่มน้ำอุ่น หลีกเลี่ยงควันบุหรี่"},
    "n": {"description": "ปอดปกติ ไม่มีความผิดปกติทางการหายใจ", "care": "รักษาสุขภาพปอดโดยการออกกำลังกายและเลี่ยงมลพิษ"},
    "copd": {"description": "โรคปอดอุดกั้นเรื้อรัง ทำให้เหนื่อยง่าย", "care": "เลิกบุหรี่ ออกกำลังกายเบา ๆ"},
    "heart failure": {"description": "หัวใจล้มเหลว ส่งผลต่อระบบหายใจ", "care": "ควบคุมอาหาร ออกกำลังกายภายใต้คำแนะนำแพทย์"},
    "lung fibrosis": {"description": "ปอดแข็งตัว สูญเสียความยืดหยุ่นในการหายใจ", "care": "เข้ารับการรักษาทางการแพทย์และหลีกเลี่ยงฝุ่นละออง"},
    "pneumonia": {"description": "ปอดอักเสบจากเชื้อโรค", "care": "พักผ่อนมาก ๆ และดื่มน้ำให้เพียงพอ"},
    "asthma": {"description": "โรคหอบหืด ทำให้หายใจติดขัด", "care": "หลีกเลี่ยงสารก่อภูมิแพ้และพกยาพ่นติดตัว"},
    "pleural effusion": {"description": "ภาวะน้ำท่วมปอด ทำให้แน่นหน้าอก", "care": "พบแพทย์เพื่อการระบายของเหลวและรักษาต้นเหตุ"}
}

# ฟังก์ชันพยากรณ์โรคจากเสียงปอด
def predict_disease(audio_file):
    audio_data, sr = librosa.load(audio_file, sr=None)
    target_length = 91875
    audio_data = librosa.util.fix_length(audio_data, size=target_length)
    
    def augment_audio(audio_data, sr, target_length=91875):
        audio_data_stretch = librosa.effects.time_stretch(audio_data, rate=1.2)
        audio_data_stretch = librosa.util.fix_length(audio_data_stretch, size=target_length)
        
        audio_data_shift = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=2)
        audio_data_shift = librosa.util.fix_length(audio_data_shift, size=target_length)
        
        noise = np.random.normal(0, 0.1, size=target_length)
        audio_data_noise = librosa.util.fix_length(audio_data, size=target_length) + noise
        
        X_augmented = np.array([audio_data, audio_data_stretch, audio_data_shift, audio_data_noise])
        return X_augmented
    
    X_augmented = augment_audio(audio_data, sr, target_length=target_length)
    X_input = np.array([X_augmented[0]])
    prediction = model.predict(X_input)
    
    class_names = ['bron', 'n', 'copd', 'heart failure', 'lung fibrosis', 'pneumonia', 'asthma', 'pleural effusion']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# UI Streamlit
st.set_page_config(page_title="วินิจฉัยโรคจากเสียงปอด", page_icon="🫁", layout="centered")
st.image("https://www.scripps.edu/_files/images/news-and-events/large-release-images/2024/20240410-schultz-bollong-lung-regeneration-920x500.jpg", width=1000)

st.title('วินิจฉัยโรคจากเสียงปอด')
st.write("อัปโหลดไฟล์เสียงจากสเตโธสโคปเพื่อวินิจฉัยโรคปอด")

uploaded_file = st.file_uploader("เลือกไฟล์เสียง (รูปแบบ WAV)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    prediction = predict_disease(uploaded_file)
    st.success(f"โรคที่ทำนายได้: **{prediction}**")
    
    if prediction in disease_info:
        st.subheader("ℹ️ ข้อมูลเกี่ยวกับโรค")
        st.write(f"**รายละเอียด:** {disease_info[prediction]['description']}")
        st.write(f"**คำแนะนำ:** {disease_info[prediction]['care']}")

st.sidebar.title("คำแนะนำ")
st.sidebar.write("- อัปโหลดไฟล์เสียงคุณภาพดีเพื่อความแม่นยำในการวินิจฉัย")