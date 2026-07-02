# ==========================================================
# SHOPPER SPECTRUM
# Customer Segmentation & Product Recommendation System
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded")
# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""<style>.main{background:#f8f9fc;}
h1,h2,h3{color:#0E4C92;}
.metric-card{
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.08);
    text-align:center;}
.sidebar .sidebar-content{
    background:#0E4C92;}
.stButton>button{
    width:100%;
    border-radius:8px;
    background:#0E4C92;
    color:white;
    font-weight:bold;}

</style>""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "C:\Users\shash\Desktop\D\Labmentix\Project 9\online_retail.csv",
        encoding="ISO-8859-1"
    )

    df.dropna(subset=["CustomerID"], inplace=True)

    df = df[df["Quantity"] > 0]

    df = df[df["UnitPrice"] > 0]

    df["CustomerID"] = df["CustomerID"].astype(int)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df

# ==========================================================
# LOAD ML MODELS
# ==========================================================

@st.cache_resource
def load_models():

    scaler = joblib.load("scaler.pkl")

    model = joblib.load("customer_segment_model.pkl")

    return scaler, model

# ==========================================================
# INITIALIZE
# ==========================================================

df = load_data()

scaler, model = load_models()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Customer Segmentation",

        "Product Recommendation"

    ]

)

# ==========================================================
# CREATE RFM TABLE
# ==========================================================

@st.cache_data
def create_rfm(df):

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        })
        .reset_index()
    )

    rfm.columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary"
    ]

    return rfm


rfm = create_rfm(df)

# ==========================================================
# BUILD CUSTOMER-PRODUCT MATRIX
# ==========================================================

@st.cache_resource
def build_recommendation_engine(df):

    customer_product = pd.pivot_table(

        df,

        index="CustomerID",

        columns="Description",

        values="Quantity",

        aggfunc="sum",

        fill_value=0

    )

    customer_product = customer_product.applymap(

        lambda x: 1 if x > 0 else 0

    )

    similarity = cosine_similarity(customer_product)

    similarity_df = pd.DataFrame(

        similarity,

        index=customer_product.index,

        columns=customer_product.index

    )

    return customer_product, similarity_df


customer_product, similarity_df = build_recommendation_engine(df)

# ==========================================================
# PRODUCT RECOMMENDATION FUNCTION
# ==========================================================

def recommend_products(customer_id, top_n=10):

    if customer_id not in similarity_df.index:

        return pd.DataFrame()

    similar_customers = (

        similarity_df[customer_id]

        .sort_values(ascending=False)

        .iloc[1:6]

        .index

    )

    purchased = set(

        customer_product.columns[

            customer_product.loc[customer_id] == 1

        ]

    )

    recommendation_score = {}

    for customer in similar_customers:

        products = customer_product.columns[

            customer_product.loc[customer] == 1

        ]

        for product in products:

            if product not in purchased:

                recommendation_score[product] = (

                    recommendation_score.get(product, 0)

                    + similarity_df.loc[customer_id, customer]

                )

    if len(recommendation_score) == 0:

        return pd.DataFrame()

    recommendations = (

        pd.DataFrame(

            recommendation_score.items(),

            columns=["Product", "Score"]

        )

        .sort_values(

            "Score",

            ascending=False

        )

        .head(top_n)

    )

    return recommendations

# ==========================================================
# KPI FUNCTION
# ==========================================================

@st.cache_data
def calculate_kpis(df):

    total_revenue = df["TotalPrice"].sum()

    total_orders = df["InvoiceNo"].nunique()

    total_customers = df["CustomerID"].nunique()

    avg_order_value = total_revenue / total_orders

    return (

        total_revenue,

        total_orders,

        total_customers,

        avg_order_value

    )


(

    total_revenue,

    total_orders,

    total_customers,

    avg_order_value

) = calculate_kpis(df)

# ==========================================================
# SEGMENT LABELS
# ==========================================================

segment_names = {

    0: "Champions",

    1: "Loyal Customers",

    2: "Potential Loyalists",

    3: "At Risk"

}

# ==========================================================
# MARKETING ACTIONS
# ==========================================================

marketing_actions = {

    "Champions":
        "Reward with loyalty programs, exclusive offers, and early access to new products.",

    "Loyal Customers":
        "Encourage repeat purchases using personalized recommendations and bundle offers.",

    "Potential Loyalists":
        "Offer discounts and targeted campaigns to increase purchase frequency.",

    "At Risk":
        "Run win-back campaigns with special coupons and personalized emails."

}

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.title("🛒 Shopper Spectrum Dashboard")

    st.markdown(
        "Analyze customer purchasing behavior, business performance, and sales trends."
    )

    st.divider()

    # ==========================
    # KPI SECTION
    # ==========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Total Revenue",
            f"${total_revenue:,.2f}"
        )

    with col2:
        st.metric(
            "🧾 Total Orders",
            f"{total_orders:,}"
        )

    with col3:
        st.metric(
            "👥 Customers",
            f"{total_customers:,}"
        )

    with col4:
        st.metric(
            "🛍 Avg Order Value",
            f"${avg_order_value:,.2f}"
        )

    st.divider()

    # ==========================
    # MONTHLY SALES TREND
    # ==========================

    monthly_sales = (
        df
        .set_index("InvoiceDate")
        .resample("M")["TotalPrice"]
        .sum()
        .reset_index()
    )

    fig_month = px.line(

        monthly_sales,

        x="InvoiceDate",

        y="TotalPrice",

        markers=True,

        title="Monthly Revenue Trend"

    )

    fig_month.update_layout(

        xaxis_title="Month",

        yaxis_title="Revenue",

        template="plotly_white"

    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )

    # ==========================
    # TOP PRODUCTS
    # ==========================

    top_products = (

        df.groupby("Description")["Quantity"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )

    fig_products = px.bar(

        top_products,

        x="Quantity",

        y="Description",

        orientation="h",

        title="Top 10 Selling Products"

    )

    fig_products.update_layout(

        template="plotly_white",

        yaxis={'categoryorder':'total ascending'}

    )

    country_sales = (

        df.groupby("Country")["TotalPrice"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )

    fig_country = px.bar(

        country_sales,

        x="Country",

        y="TotalPrice",

        title="Top Revenue by Country"

    )

    fig_country.update_layout(

        template="plotly_white"

    )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(

            fig_products,

            use_container_width=True

        )

    with right:

        st.plotly_chart(

            fig_country,

            use_container_width=True

        )

    # ==========================
    # CUSTOMER PURCHASE DISTRIBUTION
    # ==========================

    purchase_distribution = (

        rfm["Frequency"]

        .value_counts()

        .sort_index()

        .reset_index()

    )

    purchase_distribution.columns = [

        "Orders",

        "Customers"

    ]

    fig_freq = px.bar(

        purchase_distribution,

        x="Orders",

        y="Customers",

        title="Customer Purchase Frequency Distribution"

    )

    fig_freq.update_layout(

        template="plotly_white"

    )

    st.plotly_chart(

        fig_freq,

        use_container_width=True

    )

    # ==========================
    # BUSINESS INSIGHTS
    # ==========================

    st.subheader("📌 Business Insights")

    top_country = country_sales.iloc[0]["Country"]
    top_product = top_products.iloc[-1]["Description"]

    col1, col2 = st.columns(2)

    with col1:

        st.info(

            f"""
            **Top Revenue Country**

            {top_country}
            """

        )

    with col2:

        st.success(

            f"""
            **Best Selling Product**

            {top_product}
            """

        )
    # ==========================================================
# CUSTOMER SEGMENTATION
# ==========================================================

elif page == "Customer Segmentation":

    st.title("🎯 Customer Segmentation")

    st.markdown(
        "Predict a customer's segment using **Recency, Frequency and Monetary (RFM)** values."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=30,
            step=1,
            help="Days since last purchase."
        )

    with col2:
        frequency = st.number_input(
            "Frequency",
            min_value=1,
            value=5,
            step=1,
            help="Number of unique purchases."
        )

    with col3:
        monetary = st.number_input(
            "Monetary Value",
            min_value=0.0,
            value=500.0,
            step=50.0,
            help="Total customer spend."
        )

    st.write("")

    if st.button("Predict Customer Segment"):

        try:

            input_df = pd.DataFrame({

                "Recency":[recency],
                "Frequency":[frequency],
                "Monetary":[monetary]

            })

            scaled_data = scaler.transform(input_df)

            prediction = model.predict(scaled_data)[0]

            segment = segment_names.get(

                prediction,

                f"Segment {prediction}"

            )

            st.success(
                f"Predicted Segment : **{segment}**"
            )

            st.subheader("📢 Recommended Marketing Strategy")

            st.info(

                marketing_actions.get(

                    segment,

                    "No recommendation available."

                )

            )

            st.subheader("📊 Customer Profile")

            profile = pd.DataFrame({

                "Metric":[

                    "Recency",

                    "Frequency",

                    "Monetary"

                ],

                "Value":[

                    recency,

                    frequency,

                    monetary

                ]

            })

            st.dataframe(

                profile,

                use_container_width=True,

                hide_index=True

            )

            st.subheader("📖 Segment Interpretation")

            explanations = {

                "Champions":
                "High-value customers who purchase frequently and recently. Focus on retaining them with loyalty rewards.",

                "Loyal Customers":
                "Consistent buyers who respond well to personalized recommendations and membership benefits.",

                "Potential Loyalists":
                "Customers showing promise. Encourage repeat purchases through discounts and bundles.",

                "At Risk":
                "Customers who have not purchased recently. Win them back using targeted campaigns."

            }

            st.write(

                explanations.get(

                    segment,

                    "No description available."

                )

            )

        except Exception as e:

            st.error(

                f"Prediction failed: {e}"

            )

    with st.expander("ℹ Understanding RFM Metrics"):

        st.markdown("""

### **Recency**
Number of days since the customer's most recent purchase.

Lower is better.

---

### **Frequency**
How often the customer has purchased.

Higher is better.

---

### **Monetary**
Total money spent by the customer.

Higher is better.

""")

# ==========================================================
# PRODUCT RECOMMENDATION
# ==========================================================

elif page == "Product Recommendation":

    st.title("🛒 Product Recommendation System")

    st.markdown(
        """
        Recommend products using **Customer-Based Collaborative Filtering**.
        Recommendations are generated dynamically from the purchase history.
        """
    )

    st.divider()

    customer_ids = sorted(customer_product.index.tolist())

    customer_id = st.selectbox(

        "Select Customer ID",

        customer_ids

    )

    top_n = st.slider(

        "Number of Recommendations",

        min_value=5,

        max_value=20,

        value=10

    )

    st.subheader("🛍 Previously Purchased Products")

    purchased_products = customer_product.columns[
        customer_product.loc[customer_id] == 1
    ]

    if len(purchased_products):

        purchased_df = pd.DataFrame({

            "Purchased Products": purchased_products

        })

        st.dataframe(

            purchased_df,

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info("No purchase history available.")

    if st.button("Generate Recommendations"):

        with st.spinner("Finding similar customers..."):

            recommendations = recommend_products(

                customer_id,

                top_n

            )

        st.subheader("⭐ Recommended Products")

        if recommendations.empty:

            st.warning(

                "No recommendations available for this customer."

            )

        else:

            st.dataframe(

                recommendations,

                use_container_width=True,

                hide_index=True

            )

            fig = px.bar(

                recommendations,

                x="Score",

                y="Product",

                orientation="h",

                title="Recommendation Scores"

            )

            fig.update_layout(

                template="plotly_white",

                yaxis={'categoryorder':'total ascending'}

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

            st.subheader("📌 Recommendation Insights")

            st.success(

                f"""
Top {len(recommendations)} products are recommended because
customers with similar purchasing behaviour bought these items.
                """

            )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(

    """
    Shopper Spectrum | Customer Segmentation & Product Recommendation System

    Built using Streamlit • Scikit-learn • Plotly • Pandas
    """

)
