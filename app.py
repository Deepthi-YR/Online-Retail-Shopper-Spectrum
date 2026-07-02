app.py
customer_segment_model.pkl
scaler.pkl
similarity.pkl
requirements.txt

import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.write("Customer Segmentation & Product Recommendation System")

# =========================
# LOAD MODELS
# =========================
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("customer_segment_model.pkl")
    similarity_df = joblib.load("similarity.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# =========================
# CLUSTER LABELS
# =========================
segment_map = {
    0: "High Value",
    1: "Regular",
    2: "Occasional",
    3: "At Risk"
}

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend_products(product_name):

    if product_name not in similarity_df.index:
        return None

    similar = similarity_df.loc[product_name]

    recommendations = (
        similar.sort_values(ascending=False)
        .iloc[1:6]
        .index
        .tolist()
    )

    return recommendations


# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Select Module",
    [
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

# ======================================================
# CUSTOMER SEGMENTATION
# ======================================================
if menu == "Customer Segmentation":

    st.header("Customer Segmentation")

    recency = st.number_input(
        "Recency (Days)",
        min_value=0.0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0.0
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0
    )

    if st.button("Predict Cluster"):

        input_data = [[recency, frequency, monetary]]

        scaled = scaler.transform(input_data)

        cluster = model.predict(scaled)[0]

        st.success(
            f"Customer Segment: {segment_map.get(cluster, cluster)}"
        )

# ======================================================
# PRODUCT RECOMMENDATION
# ======================================================
elif menu == "Product Recommendation":

    st.header("Product Recommendation")

    product = st.text_input(
        "Enter Product Name"
    )

    if st.button("Get Recommendations"):

        recommendations = recommend_products(product)

        if recommendations is None:

            st.error("Product not found.")

        else:

            st.success("Recommended Products")

            for item in recommendations:

                st.write("✅", item)
