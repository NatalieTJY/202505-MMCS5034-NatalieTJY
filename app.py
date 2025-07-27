import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import tempfile
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random
import re

# Remove Emoji Function
def strip_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Streamlit UI
st.title("🤖 CrackGuard – Your first line of defence against failure")
st.text("created by NatalieTJY")

# Load trained model once
@st.cache_resource
def load_model():
    return joblib.load("trained_model_LogisticRegression.pkl")

model = load_model()

# Sidebar: Sitcom Selection
st.sidebar.header("🎬 Sitcom Corner")
sitcom = st.sidebar.selectbox("Choose a sitcom:", ["Friends", "The Big Bang Theory", "The Office"])
sitcom_mode = st.sidebar.checkbox("✨ Activate Sitcom Mode")

# Sitcom Assets
quotes = {
    "Friends": ["We were on a break!", "Pivot! Pivot!"],
    "The Big Bang Theory": ["Bazinga!", "I'm not insane. My mother had me tested."],
    "The Office": ["I'm not superstitious, but I am a little stitious.", "Bears. Beets. Battlestar Galactica."]
}

st.sidebar.markdown(f"**Random Sitcom Quote:** {random.choice(quotes[sitcom])}")

# Preprocessing function
def preprocess_image(img):
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)  # Fix: Use RGB2GRAY
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        resized = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)
        return resized.flatten().reshape(1, -1)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None

# Upload Images
uploaded_files = st.file_uploader("Upload multiple images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
results = []

# Sitcom Reaction Content 
reaction_quotes = {
    "Positive": {
        "Friends": "Joey: 'How you doin'?' 😏",
        "The Big Bang Theory": "Sheldon: 'Bazinga!' 💥",
        "The Office": "Michael: 'The greatest day of all time.' 🎉"
    },
    "Negative": {
        "Friends": "Ross: 'We were on a break...' 😩",
        "The Big Bang Theory": "Amy: 'No.' 😐",
        "The Office": "Dwight: 'False.' 😒"
    }
}

reaction_gifs = {
    "Positive": {
        "Friends": "friends_positive.gif",
        "The Big Bang Theory": "tbbt_positive.gif",
        "The Office": "office_positive.gif"
    },
    "Negative": {
        "Friends": "friends_negative.gif",
        "The Big Bang Theory": "tbbt_negative.gif",
        "The Office": "office_negative.gif"
    }
}

theme_audio = {
    "Friends": "'I'll Be There For You' (Friends Theme Song).mp3",
    "The Big Bang Theory": "The Big Bang Theory Theme Song.mp3",
    "The Office": "The Office Theme Song.mp3"
}

# Prediction Phase
if uploaded_files:
    st.subheader("🖼️ Image Predictions")
    
    with st.spinner("Processing images..."):
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Cannot open {uploaded_file.name}: {e}")
                continue

            st.image(image, caption=uploaded_file.name, use_container_width =True)

            features = preprocess_image(image)
            if features is None:
                continue

            try:
                prediction = model.predict(features)[0]
                label = "Positive" if prediction == 1 else "Negative"

                # Optional: Confidence score
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)[0]
                    confidence = proba[prediction]
                    st.markdown(f"**Prediction for {uploaded_file.name}:** {label} ({confidence:.2f} confidence)")
                else:
                    st.markdown(f"**Prediction for {uploaded_file.name}:** {label}")

                results.append((image, uploaded_file.name, label))

                # Show reaction only if Sitcom Mode is activated
                if sitcom_mode:
                    st.markdown(f"### 🎬 Sitcom Reaction")
                    st.markdown(f"**{reaction_quotes[label][sitcom]}**")
                    st.image(reaction_gifs[label][sitcom], caption=f"{sitcom} Reaction")
                    st.markdown(f"🎵 *Enjoy the theme song from {sitcom}*")
                    st.audio(theme_audio[sitcom])

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    # Pie chart
    st.subheader("📊 Overall Prediction Summary")
    labels_list = [label.split()[0] for _, _, label in results]
    positives = labels_list.count("Positive")
    negatives = labels_list.count("Negative")

    fig, ax = plt.subplots()

    chart_colors = {"Friends": ["#9ACD32", "#DC143C"], "The Big Bang Theory": ["#FFD700", "#4682B4"], "The Office": ["#4169E1", "#2F4F4F"]}

    ax.pie([positives, negatives], labels=["Positive", "Negative"], autopct='%1.1f%%', colors=chart_colors.get(sitcom, ["green", "red"]))

    ax.axis("equal")
    st.pyplot(fig)

    # Save pie chart to disk
    pie_path = "pie_chart.png"
    fig.savefig(pie_path)

    # PDF Generation
    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Add Pie Chart
            pdf.add_page()

            title_map = {
                "Friends": "The One With the Predictions",
                "The Big Bang Theory": "The Predictive Entanglement",
                "The Office": "That's What the Model Said"
            }

            pdf.set_font("Helvetica", size=14)
            pdf.cell(200, 10, title_map[sitcom], ln=True, align='C')
            pdf.set_font("Helvetica", size=10)
            pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.image(pie_path, x=10, y=30, w=180)

            # Add images and predictions
            for img, name, label in results:
                pdf.add_page()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                        img.save(tmpfile.name, "JPEG")
                        pdf.set_font("Helvetica", size=12)
                        pdf.cell(200, 10, f"Image: {name}", ln=True)
                        pdf.cell(200, 10, f"Prediction: {label}", ln=True)
                        
                        clean_quote = strip_emojis(reaction_quotes[label][sitcom])
                        pdf.multi_cell(0, 10, f"Sitcom Quote: {clean_quote}")

                        pdf.image(tmpfile.name, x=10, y=50, w=80)
                finally:
                    if os.path.exists(tmpfile.name):
                        os.remove(tmpfile.name)  # Clean up temp file

            # Convert to BytesIO for download
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_output = io.BytesIO(pdf_bytes)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_output,
                file_name=f"prediction_report_{timestamp}.pdf",
                mime="application/pdf"
            )

        # Clean up pie chart image
    if os.path.exists(pie_path):
        os.remove(pie_path)
