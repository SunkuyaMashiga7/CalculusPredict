import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import shap
import google.generativeai as genai
import config  # ดึงค่า API Key จากไฟล์ config.py

# ==========================================
# 1. ส่วนเตรียมข้อมูลและโมเดลจำลอง (Mock Data & Model)
# ==========================================
@st.cache_resource 
def load_mock_model():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'math_base': np.random.randint(1, 6, n),
        'miss_class': np.random.randint(1, 5, n),
        'practice': np.random.randint(1, 5, n),
        'lack_sleep': np.random.randint(1, 5, n)
    })
    risk_score = (df['miss_class']*1.5) + (df['lack_sleep']*1.2) - (df['practice']*2) - df['math_base']
    df['target_f'] = (risk_score > 0).astype(int)

    X = df.drop('target_f', axis=1)
    y = df['target_f']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X

model, X_train = load_mock_model()

# ==========================================
# 2. การตั้งค่าหน้าจอ (UI Setup)
# ==========================================
st.set_page_config(page_title="Calculus Grade Predictor", layout="wide")

st.title("📊 Calculus Risk Predictor & AI Assistant")
st.markdown("ประเมินความเสี่ยงผลการเรียนวิชาแคลคูลัส พร้อมคำแนะนำจาก Explainable AI และ Gemini")

# ==========================================
# 3. ส่วนหน้าจอ UI หลัก (User Interface)
# ==========================================
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 แบบประเมินพฤติกรรม (1 เดือนที่ผ่านมา)")
    math_base = st.slider("ระดับพื้นฐานคณิตศาสตร์ของคุณ (1=อ่อน, 5=เชี่ยวชาญ)", 1, 5, 3)
    
    # สร้างตัวแปรเก็บข้อความอธิบายตัวเลือก
    freq_options = {
        1: "1 (ไม่เคยเลย)",
        2: "2 (น้อยกว่าสัปดาห์ละครั้ง)",
        3: "3 (1-2 ครั้ง/สัปดาห์)",
        4: "4 (3 ครั้งขึ้นไป)"
    }
    
    # ใช้ format_func เพื่อนำตัวเลขไปแมปกับข้อความอธิบายตอนแสดงผล
    miss_class = st.selectbox(
        "คุณขาดคลาสเรียนวิชาแคลคูลัสบ่อยแค่ไหน?", 
        options=list(freq_options.keys()), 
        format_func=lambda x: freq_options[x]
    )
    
    practice = st.selectbox(
        "คุณฝึกทำโจทย์แคลคูลัสนอกเวลาเรียนบ่อยแค่ไหน?", 
        options=list(freq_options.keys()), 
        format_func=lambda x: freq_options[x]
    )
    
    lack_sleep = st.selectbox(
        "คุณนอนหลับน้อยกว่า 6 ชั่วโมงต่อวันบ่อยแค่ไหน?", 
        options=list(freq_options.keys()), 
        format_func=lambda x: freq_options[x]
    )
    
    predict_btn = st.button("🚀 ประเมินความเสี่ยง", use_container_width=True)

# ==========================================
# 4. ส่วนประมวลผลและแสดงกราฟ (Prediction & XAI Dashboard)
# ==========================================
if predict_btn:
    with col2:
        st.subheader("🎯 ผลการวิเคราะห์ของคุณ")
        
        input_data = pd.DataFrame({
            'math_base': [math_base],
            'miss_class': [miss_class],
            'practice': [practice],
            'lack_sleep': [lack_sleep]
        })
        
        prob = model.predict_proba(input_data)[0]
        risk_prob = prob[1] * 100 
        safe_prob = prob[0] * 100
        
        fig_pie = px.pie(
            values=[risk_prob, safe_prob], 
            names=['โอกาสติด F (เสี่ยง)', 'โอกาสผ่าน (ปลอดภัย)'],
            color_discrete_sequence=['#FF4B4B', '#00CC96'],
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        if isinstance(shap_values, list):
            shap_vals_fail = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            shap_vals_fail = shap_values[0, :, 1]
        else:
            shap_vals_fail = shap_values[0]
            
        feature_names = ['พื้นฐานคณิตศาสตร์', 'การขาดเรียน', 'การทำโจทย์', 'การนอนน้อย']
        
        df_shap = pd.DataFrame({'ปัจจัย': feature_names, 'ผลกระทบ': shap_vals_fail})
        df_shap['สี'] = df_shap['ผลกระทบ'].apply(lambda x: 'เพิ่มความเสี่ยง' if x > 0 else 'ลดความเสี่ยง')
        
        fig_shap = px.bar(
            df_shap, x='ผลกระทบ', y='ปัจจัย', orientation='h', 
            color='สี', color_discrete_map={'เพิ่มความเสี่ยง': '#FF4B4B', 'ลดความเสี่ยง': '#00CC96'},
            title="🔍 ปัจจัยที่ส่งผลต่อคะแนนของคุณ (XAI Analysis)"
        )
        fig_shap.update_layout(yaxis={'categoryorder':'total ascending'}, height=300)

        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with subcol2:
            st.plotly_chart(fig_shap, use_container_width=True)
            
        st.session_state['risk_prob'] = risk_prob
        st.session_state['xai_context'] = df_shap.to_dict('records')

# ==========================================
# 5. ส่วนผู้ช่วย AI ส่วนตัว (Gemini Chat Integration)
# ==========================================
st.divider()
st.subheader("💬 Gemini AI Assistant (ที่ปรึกษาด้านการเรียน)")

def initialize_gemini():
    api_key = config.GEMINI_API_KEY
    if api_key:
        genai.configure(api_key=api_key)
        
        try:
            # ใช้โมเดล gemini-3.8-flash ตามที่ระบบของ Google แนะนำล่าสุด
            model_gemini = genai.GenerativeModel('gemini-3.6-flash')
            
            if "chat_session" not in st.session_state:
                initial_history = [
                    {"role": "user", "parts": ["จากนี้ไปคุณคือ AI ผู้ช่วยที่ปรึกษาด้านการเรียนวิชาแคลคูลัส คุณมีหน้าที่ให้คำแนะนำเชิงบวก เข้าใจง่าย และให้กำลังใจนักศึกษานะครับ"]},
                    {"role": "model", "parts": ["รับทราบครับ! ผมพร้อมทำหน้าที่เป็นที่ปรึกษาด้านการเรียนวิชาแคลคูลัส เพื่อให้คำแนะนำและเป็นกำลังใจให้นักศึกษาทุกคนแล้วครับ มีข้อมูลหรือคำถามอะไรให้ผมช่วยวิเคราะห์ ส่งมาได้เลยครับ"]}
                ]
                st.session_state.chat_session = model_gemini.start_chat(history=initial_history)
            return True
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Google API: {e}")
            return False
            
    return False

is_api_ready = initialize_gemini()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "สวัสดีครับ! ลองประเมินความเสี่ยงด้านบน แล้วพิมพ์ปรึกษาวิธีเรียนกับผมได้เลยครับ"})

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("ปรึกษา Gemini: ทำยังไงให้เกรดแคลคูลัสดีขึ้น?"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if not is_api_ready:
        error_msg = "⚠️ ไม่พบ API Key ในไฟล์ config.py กรุณาตรวจสอบไฟล์อีกครั้งครับ"
        st.chat_message("assistant").write(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        if 'risk_prob' in st.session_state and 'xai_context' in st.session_state:
            enriched_prompt = f"ข้อมูลของนักศึกษา: ความเสี่ยงติด F คือ {st.session_state['risk_prob']:.0f}%\n"
            enriched_prompt += f"ปัจจัยที่มีผล: {st.session_state['xai_context']}\n"
            enriched_prompt += f"คำถามจากนักศึกษา: {prompt}\n(กรุณาตอบกลับเฉพาะคำถามของนักศึกษา โดยนำข้อมูลข้างต้นมาช่วยแนะนำให้ตรงจุด)"
            response = st.session_state.chat_session.send_message(enriched_prompt)
        else:
            response = st.session_state.chat_session.send_message(prompt)
            
        st.chat_message("assistant").write(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})