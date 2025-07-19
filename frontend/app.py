'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This Streamlit web application performs deepfake image detection using multiple trained models.
// The system includes:
// 1) Image upload, preview, and preprocessing
// 2) Model prediction aggregation (Real/Fake)
// 3) Confidence visualization through charts (bar, pie, line)
// 4) Final verdict generation based on majority voting
// 5) Report export in PDF and XML formats with download options
//
// Remarks:
// - Uses predict_deepfake() from predict.py to generate per-model results
// - Supports JPEG/PNG uploads
// - Outputs are downloadable and visually interpreted
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

import streamlit as st
from PIL import Image
import os
import sys
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import xmltodict

# Custom import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict import predict_deepfake

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Page setup
st.set_page_config(page_title="DeepFake Detector", layout="centered", page_icon="🧠")
st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🧠 Advanced DeepFake Detection System</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# 📘 INTRODUCTION CONTENT (Always Visible — Not in Dropdown)
# ───────────────────────────────────────────────────────────────

st.markdown("""
### 🤖 What is DeepFake?
DeepFakes are AI-generated synthetic media where someone's face or voice is replaced with someone else's.  
They can be realistic enough to deceive people and are increasingly used for misinformation, fraud, and media manipulation.

### 🎯 What This System Does:
- Accepts a face image and passes it through **multiple trained deep learning models**
- Compares predictions from each model to generate a **final verdict**
- Provides **visual insights** using charts
- Generates downloadable **PDF** and **XML reports**

---

### 🧪 How Models Work:
Each model outputs:
- A **label** (`Real` or `Fake`)
- A **confidence score** representing how sure the model is

We take a **majority vote** and also consider the **average confidence** to determine the final verdict.

""")

# ───────────────────────────────────────────────────────────────
# 📤 IMAGE UPLOAD
# ───────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("📤 Upload a Face Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    st.markdown("### 🖼️ Uploaded Image Preview")
    image = Image.open(uploaded_file)
    st.image(image, caption="Analyzing this image...", use_column_width=True)

    # Save image locally
    saved_path = os.path.join("uploads", "temp.jpg")
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Animate analysis
    progress_text = "🔍 Running predictions using multiple AI models..."
    my_bar = st.progress(0)
    with st.spinner(progress_text):
        results = predict_deepfake(saved_path)
        my_bar.progress(100)

    # ───────────────────────────────────────────────────────────────
    # 🔍 PROCESS RESULTS
    # ───────────────────────────────────────────────────────────────

    data = []
    real_votes = 0
    for model_name, info in results.items():
        label = info['label']
        confidence = info['confidence']
        final_conf = 100 - confidence * 100 if label.lower() == 'fake' else confidence * 100
        real_votes += 1 if label.lower() == 'real' else 0
        data.append({
            "Model": model_name,
            "Label": label,
            "Confidence (%)": round(final_conf, 2)
        })

    df = pd.DataFrame(data)

    # ───────────────────────────────────────────────────────────────
    # 🎯 FINAL VERDICT
    # ───────────────────────────────────────────────────────────────

    avg_confidence = df["Confidence (%)"].mean()
    final_label = "Real" if real_votes >= len(data)/2 else "Fake"
    final_color = "green" if final_label == "Real" else "red"

    st.markdown("## 📊 Model-wise Predictions")
    for row in data:
        emoji = "✅" if row['Label'].lower() == 'real' else "❌"
        st.markdown(f"{emoji} **{row['Model']}** ➤ `{row['Label'].upper()}` (**{row['Confidence (%)']}%** confidence)")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <h2 style='text-align:center; color:{final_color};'>
        🎯 Final Verdict: {final_label.upper()}<br>
        📈 Average Confidence: {avg_confidence:.2f}%
    </h2>
    """, unsafe_allow_html=True)

    # ───────────────────────────────────────────────────────────────
    # 📊 CONFIDENCE COMPARISON BAR CHART
    # ───────────────────────────────────────────────────────────────

    st.markdown("### 📈 Confidence Comparison")
    fig_bar = px.bar(
        df,
        x="Model",
        y="Confidence (%)",
        color="Label",
        text="Confidence (%)",
        color_discrete_map={"real": "green", "fake": "red"},
        height=400
    )
    fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_bar.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_bar, use_container_width=True)

    # ───────────────────────────────────────────────────────────────
    # 🧩 PIE CHART with Confidence Weights
    # ───────────────────────────────────────────────────────────────

    custom_colors = px.colors.qualitative.Bold  # or use Safe, D3, Pastel, etc.

    # 🧩 PIE CHART with Distinct Section Colors
    st.markdown("### 🧩 Confidence Contribution Pie Chart (by Model)")
    fig_pie = px.pie(
        df,
        values="Confidence (%)",
        names="Model",
        title="Model-wise Confidence Contribution",
        color="Model",  # Use Model name to color differently
        color_discrete_sequence=custom_colors,
        hole=0.3  # Optional: for a donut chart look
    )
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        pull=[0.05] * len(df),  # Slight pull effect for emphasis
    )
    fig_pie.update_layout(
        legend_title="Model",
        font=dict(size=14),
        title_font=dict(size=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ───────────────────────────────────────────────────────────────
    # 📉 LINE CHART for Trend
    # ───────────────────────────────────────────────────────────────

    st.markdown("### 📉 Model Confidence Trend")
    fig_line = px.line(df, x='Model', y='Confidence (%)', markers=True, title="Confidence Across Models")
    st.plotly_chart(fig_line)

    # ───────────────────────────────────────────────────────────────
    # 📐 STATISTICAL SUMMARY
    # ───────────────────────────────────────────────────────────────

    st.markdown("### 📐 Statistical Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Confidence", f"{df['Confidence (%)'].mean():.2f}%")
    col2.metric("Highest Confidence", f"{df['Confidence (%)'].max():.2f}%")
    col3.metric("Lowest Confidence", f"{df['Confidence (%)'].min():.2f}%")

    # ───────────────────────────────────────────────────────────────
    # 📋 RAW TABLE VIEW
    # ───────────────────────────────────────────────────────────────

    st.markdown("### 📋 Prediction Data Table")
    st.dataframe(df.set_index("Model"), use_container_width=True)

    # ───────────────────────────────────────────────────────────────
    # 📝 REPORT GENERATION
    # ───────────────────────────────────────────────────────────────

    # PDF
    pdf_path = "reports/report.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="DeepFake Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Final Verdict: {final_label} (Avg. Confidence: {avg_confidence:.2f}%)", ln=True)
    pdf.ln(5)
    for row in data:
        pdf.cell(200, 10, txt=f"{row['Model']}: {row['Label']} ({row['Confidence (%)']}%)", ln=True)
    pdf.output(pdf_path)

    # XML
    xml_data = {
        "DeepFakeReport": {
            "Image": uploaded_file.name,
            "FinalVerdict": {
                "Label": final_label,
                "AvgConfidence": round(avg_confidence, 2)
            },
            "Models": {row["Model"]: {"Label": row["Label"], "Confidence": row["Confidence (%)"]} for row in data}
        }
    }
    xml_path = "reports/report.xml"
    with open(xml_path, "w") as f:
        f.write(xmltodict.unparse(xml_data, pretty=True))

    # ───────────────────────────────────────────────────────────────
    # 📥 DOWNLOAD BUTTONS
    # ───────────────────────────────────────────────────────────────

    st.markdown("### 📥 Download Your Report")
    col_pdf, col_xml = st.columns(2)
    with open(pdf_path, "rb") as pdf_file:
        col_pdf.download_button("📄 Download PDF Report", data=pdf_file, file_name="deepfake_report.pdf", mime="application/pdf")
    with open(xml_path, "rb") as xml_file:
        col_xml.download_button("🧾 Download XML Report", data=xml_file, file_name="deepfake_report.xml", mime="application/xml")

    st.success("✅ All predictions, stats, and reports are ready.")
    st.progress(100)
