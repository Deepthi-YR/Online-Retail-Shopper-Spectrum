
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="SHOPPER SPECTRUM",
    layout="wide"
)

st.title("🍲 Shopper Spectrum")


menu = st.sidebar.radio(
    "Choose",
    [
        "Dashboard",
        "Customer Segmentation",
        "Product Recommendation"
    ]
)


#----------Module 1: Product recommendation----------
st.header("Product Recommendation")
product = st.text_input(
    "Enter Product Name"
)
if st.button("Get Recommendations"):
for item in recommendations:
    st.success(item)

#----------Module 2: Customer segmentaion-------------

st.header("Customer Segmentation")
recency = st.number_input(
    "Recency"
)

frequency = st.number_input(
    "Frequency"
)

monetary = st.number_input(
    "Monetary"
)

scaled = scaler.transform(
    [[recency,frequency,monetary]]
)

cluster = model.predict(
    scaled
)
