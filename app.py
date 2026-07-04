import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🛍️ Shopper Spectrum")
st.subheader("Customer Segmentation using RFM Analysis & Machine Learning")

st.markdown("---")

# -----------------------------
# Dataset Loading
# -----------------------------
DATA_PATH = "C:/Users/shash/Desktop/D/Labmentix/Project 9/online_retail.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    return df

try:
    df = load_data()
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -----------------------------
# Dataset Preview
# -----------------------------
st.header("Dataset Preview")

st.dataframe(df.head())

st.markdown("---")

# -----------------------------
# Dataset Information
# -----------------------------
st.header("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.markdown("---")

# -----------------------------
# Basic KPIs
# -----------------------------
st.header("Business KPIs")

# Create Total Amount
if "TotalAmount" not in df.columns:
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

total_customers = df["CustomerID"].nunique()
total_orders = df["InvoiceNo"].nunique()
total_revenue = df["TotalAmount"].sum()
avg_order = total_revenue / total_orders

c1, c2, c3, c4 = st.columns(4)

c1.metric("Customers", f"{total_customers:,}")
c2.metric("Orders", f"{total_orders:,}")
c3.metric("Revenue", f"£{total_revenue:,.2f}")
c4.metric("Average Order Value", f"£{avg_order:,.2f}")

st.markdown("---")

# -----------------------------
# Raw Data
# -----------------------------
if st.checkbox("Show Complete Dataset"):
    st.dataframe(df)
