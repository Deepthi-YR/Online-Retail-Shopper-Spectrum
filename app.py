import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------------------------
# Load Files
# -------------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    cluster_labels = joblib.load("cluster_labels.pkl")
    item_similarity = joblib.load("item_similarity.pkl")

    return scaler, kmeans, cluster_labels, item_similarity


@st.cache_data
def load_data():
    return pd.read_csv("cleaned_online_retail.csv")


try:
    scaler, kmeans, cluster_labels, item_similarity = load_models()
    df = load_data()

except Exception as e:
    st.error(f"Error loading files:\n\n{e}")
    st.stop()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Product Recommendation",
        "Customer Segmentation"
    ]
)

# -------------------------------------------------
# HOME
# -------------------------------------------------
if page == "Home":

    st.title("🛒 Shopper Spectrum")

    st.subheader("Customer Segmentation & Product Recommendation")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Transactions", len(df))

    with col2:
        st.metric("Customers", df["CustomerID"].nunique())

    with col3:
        st.metric("Products", df["Description"].nunique())

    st.markdown("---")

    st.header("Project Overview")

    st.write("""
This application performs:

- Customer Segmentation using RFM Analysis and KMeans Clustering
- Product Recommendation using Item-Based Collaborative Filtering
- Real-time prediction using trained machine learning models
""")

# -------------------------------------------------
# PRODUCT RECOMMENDATION
# -------------------------------------------------
elif page == "Product Recommendation":

    st.title("🛍 Product Recommendation")

    product_name = st.text_input("Enter Product Name")

    if st.button("Recommend"):

        if product_name.strip() == "":
            st.warning("Please enter a product name.")

        else:

            product_upper = item_similarity.index.str.upper()

            if product_name.upper() in product_upper:

                original_name = item_similarity.index[
                    product_upper.get_loc(product_name.upper())
                ]

                recommendations = (
                    item_similarity[original_name]
                    .sort_values(ascending=False)
                    .iloc[1:6]
                    .index
                )

                st.success("Top 5 Recommended Products")

                for product in recommendations:
                    st.write("✅", product)

            else:
                st.error("Product not found.")

# -------------------------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------------------------
elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    recency = st.number_input(
        "Recency (Days)",
        min_value=0.0,
        value=30.0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0.0,
        value=5.0
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Cluster"):

        sample = scaler.transform([[recency, frequency, monetary]])

        cluster = kmeans.predict(sample)[0]

        segment = cluster_labels.get(cluster, "Unknown")

        st.success(f"Customer Segment: {segment}")
