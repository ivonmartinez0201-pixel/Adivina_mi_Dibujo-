import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Adivina mi Dibujo - Proyecto IA",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #000000 !important;
    }
    .stCanvas, canvas, .stCanvas > div, [data-testid="stCanvas"] {
        background-color: #000000 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FFA500, #FF6B00) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 25px !important;
        transition: 0.3s !important;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(255,165,0,0.4);
    }
    .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
        background-color: transparent !important;
        color: #000000 !important;
        border: none !important;
    }
    .stAlert p, .stInfo p {
        color: #000000 !important;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
    }
    .category-card {
        border-radius: 12px;
        padding: 10px;
        margin: 5px 0;
        text-align: center;
        transition: 0.3s;
        cursor: default;
    }
    .category-card:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(255,165,0,0.3);
    }
    .category-card .emoji {
        font-size: 32px;
        display: block;
    }
    .category-card .name {
        color: #000000;
        font-size: 14px;
        font-weight: 500;
    }
    .prediction-box {
        background: rgba(0,0,0,0.05);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        border: 2px solid rgba(0,0,0,0.1);
        text-align: center;
    }
    .confidence-bar-container {
        background: rgba(0,0,0,0.1);
        border-radius: 20px;
        height: 30px;
        margin: 8px 0;
        overflow: hidden;
        position: relative;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 20px;
        transition: width 0.5s ease;
        background: linear-gradient(90deg, #FFA500, #FF6B00);
    }
    .confidence-bar-text {
        position: absolute;
        right: 12px;
        top: 50%;
        transform: translateY(-50%);
        color: white;
        font-weight: bold;
        font-size: 13px;
        text-shadow: 0 0 5px rgba(0,0,0,0.5);
    }
    .result-emoji {
        font-size: 96px;
        line-height: 1.2;
        display: block;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    try:
        model = load_model('modelo_dibujos.h5')
        return model
    except Exception as e:
        st.error(f"❌ No se encontró el modelo. Ejecuta train.py primero.\nError: {e}")
        return None

model = load_trained_model()

categories = [
    'cake', 'bus', 'cell phone', 'circle',
    'crayon', 'door', 'eye', 'fish',
    'flower', 'hexagon', 'donut', 'triangle'
]

category_emojis = {
    'cake': '🎂', 'bus': '🚌', 'cell phone': '📱', 'circle': '⭕',
    'crayon': '🖍️', 'door': '🚪', 'eye': '👁️', 'fish': '🐟',
    'flower': '🌸', 'hexagon': '⬡', 'donut': '🍩', 'triangle': '🔺'
}

category_colors = {
    'cake': '#FF6B6B', 'bus': '#FDCB6E', 'cell phone': '#6C5CE7', 'circle': '#00B894',
    'crayon': '#E17055', 'door': '#0984E3', 'eye': '#2D3436', 'fish': '#00CEC9',
    'flower': '#FD79A8', 'hexagon': '#A29BFE', 'donut': '#FDCB6E', 'triangle': '#FFD93D'
}

st.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h1 style="font-size: 48px; background: linear-gradient(135deg, #FFA500, #FF6B00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Adivina mi Dibujo
    </h1>
    <p style="color: #555; font-size: 18px;">Dibuja algo y la IA adivinará lo que es</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 📋 Categorías disponibles")
cols = st.columns(6)
for i, cat in enumerate(categories):
    with cols[i % 6]:
        st.markdown(f"""
        <div class="category-card" style="background: {category_colors[cat]}22; border: 2px solid {category_colors[cat]};">
            <span class="emoji">{category_emojis[cat]}</span>
            <span class="name">{cat.capitalize()}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

col_canvas, col_results = st.columns([2, 1])

with col_canvas:
    st.subheader("✏️ Lienzo")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("📊 Ver Matriz de Confusión", use_container_width=True):
        try:
            from PIL import Image as PILImage
            img = PILImage.open('confusion_matrix.png')
            st.image(img, caption="Matriz de Confusión", use_container_width=True)
        except:
            st.warning("Ejecuta train.py primero para generar la matriz")

with col_results:
    st.subheader("📊 Predicción en Tiempo Real")
    pred_placeholder = st.empty()
    confidence_placeholder = st.empty()
    stats_placeholder = st.empty()
    debug_placeholder = st.empty()

if model is not None and canvas_result.image_data is not None:
    img = canvas_result.image_data.astype('uint8')
    img_gray = Image.fromarray(img).convert('L')
    img_array_gray = np.array(img_gray)
    has_drawing = np.any(img_array_gray > 50)

    if has_drawing:
        img_pil = Image.fromarray(img).convert('L').resize((28, 28))
        img_array = np.array(img_pil).astype('float32') / 255.0
        img_array = 1 - img_array
        img_array = img_array.reshape(1, 28, 28, 1)
        prediction = model.predict(img_array, verbose=0)[0]

        top_3_idx = np.argsort(prediction)[-3:][::-1]
        top_3_probs = prediction[top_3_idx]
        top_3_cats = [categories[i] for i in top_3_idx]

        best_cat = top_3_cats[0]
        best_prob = top_3_probs[0]

        if best_prob > 0.75:
            bg_color = category_colors[best_cat]
            confidence_level = "Alta 🟢"
        elif best_prob > 0.50:
            bg_color = "#FFA500"
            confidence_level = "Media 🟡"
        else:
            bg_color = "#FF4444"
            confidence_level = "Baja 🔴"

        with pred_placeholder.container():
            st.markdown(f"""
            <div class="prediction-box" style="border-color: {category_colors[best_cat]}40;">
                <span class="result-emoji">{category_emojis[best_cat]}</span>
                <h2 style="color: {category_colors[best_cat]}; margin: 0;">{best_cat.capitalize()}</h2>
                <p style="font-size: 20px; color: #333; margin: 5px 0;">Confianza: <strong>{best_prob*100:.1f}%</strong></p>
                <p style="font-size: 16px; color: {bg_color};">Nivel de confianza: {confidence_level}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.write("### 🔝 Top 3 predicciones:")
            for i, (cat, prob) in enumerate(zip(top_3_cats, top_3_probs)):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 8px 0; padding: 8px; background: {category_colors[cat]}11; border-radius: 8px; border-left: 5px solid {category_colors[cat]};">
                    <span style="font-size: 24px; margin-right: 10px;">{emoji}</span>
                    <span style="font-size: 18px; width: 80px; font-weight: bold; color: #000;">{cat.capitalize()}</span>
                    <div class="confidence-bar-container" style="flex: 1;">
                        <div class="confidence-bar-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {category_colors[cat]}, {category_colors[cat]}cc);"></div>
                        <span class="confidence-bar-text">{prob*100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        confidence_placeholder.progress(
            float(best_prob),
            text=f"🎯 Confianza de la mejor predicción: {best_prob*100:.1f}%"
        )

        with stats_placeholder.container():
            st.markdown("---")
            st.write("### 📊 Estadísticas")
            col_stats = st.columns(3)
            with col_stats[0]:
                st.metric("Categoría", f"{best_cat.capitalize()}")
            with col_stats[1]:
                st.metric("Confianza", f"{best_prob*100:.1f}%")
            with col_stats[2]:
                if len(top_3_probs) > 1:
                    diff = (top_3_probs[0] - top_3_probs[1]) * 100
                    st.metric("Diferencia", f"{diff:.1f}%")
                else:
                    st.metric("Diferencia", "N/A")

        with debug_placeholder.expander("🔍 Ver imagen procesada (28x28)"):
            st.image(img_array.reshape(28, 28), width=150, clamp=True)

    else:
        pred_placeholder.info("✏️ Dibuja algo en el lienzo para comenzar...")
        confidence_placeholder.empty()
        stats_placeholder.empty()
        debug_placeholder.empty()

else:
    if model is None:
        st.warning("⚠️ Modelo no cargado. Ejecuta `python train.py` primero.")
    else:
        st.info("✏️ Dibuja algo en el lienzo para comenzar...")