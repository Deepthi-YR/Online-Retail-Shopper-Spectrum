# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="Shopper Spectrum", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding="latin1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["CustomerID","Description"])
    df["CustomerID"]=df["CustomerID"].astype(int)
    df["Sales"]=df["Quantity"]*df["UnitPrice"]
    return df

@st.cache_resource
def load_models():
    return joblib.load("scaler.pkl"), joblib.load("customer_segment_model.pkl")

@st.cache_resource
def build_recommender(df):
    basket=(df.groupby(["CustomerID","Description"])["Quantity"]
              .sum().unstack(fill_value=0))
    sim=cosine_similarity(basket)
    sim=pd.DataFrame(sim,index=basket.index,columns=basket.index)
    return basket,sim

df=load_data()
scaler,model=load_models()
basket,sim=build_recommender(df)

st.sidebar.title("Shopper Spectrum")
page=st.sidebar.radio("Navigation",["Dashboard","Customer Segmentation","Product Recommendation"])

if page=="Dashboard":
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Revenue",f"${df['Sales'].sum():,.0f}")
    c2.metric("Customers",df.CustomerID.nunique())
    c3.metric("Orders",df.InvoiceNo.nunique())
    c4.metric("Products",df.Description.nunique())

    st.subheader("Top Products")
    top=df.groupby("Description")["Sales"].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(top,x="Sales",y="Description",orientation="h"),use_container_width=True)

    st.subheader("Monthly Revenue")
    m=df.assign(Month=df.InvoiceDate.dt.to_period("M").astype(str)).groupby("Month")["Sales"].sum().reset_index()
    st.plotly_chart(px.line(m,x="Month",y="Sales"),use_container_width=True)

elif page=="Customer Segmentation":
    st.subheader("Predict Customer Segment")
    r=st.number_input("Recency",1,1000,30)
    f=st.number_input("Frequency",1,500,5)
    m=st.number_input("Monetary",0.0,1000000.0,500.0)
    if st.button("Predict"):
        X=scaler.transform([[r,f,m]])
        seg=model.predict(X)[0]
        st.success(f"Predicted Segment: {seg}")

else:
    st.subheader("Product Recommendation")
    ids=sorted(basket.index.tolist())
    cid=st.selectbox("Customer",ids)
    if st.button("Recommend"):
        neigh=sim.loc[cid].sort_values(ascending=False).iloc[1:6].index
        owned=set(basket.columns[basket.loc[cid]>0])
        scores={}
        for n in neigh:
            for p in basket.columns[basket.loc[n]>0]:
                if p not in owned:
                    scores[p]=scores.get(p,0)+basket.loc[n,p]
        if scores:
            rec=pd.DataFrame(scores.items(),columns=["Product","Score"]).sort_values("Score",ascending=False).head(10)
            st.dataframe(rec,use_container_width=True)
        else:
            st.info("No recommendations available.")
