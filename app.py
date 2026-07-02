#SIDEBAR
menu = st.sidebar.radio(
    "Choose",
    [
        "Dashboard",
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

Module 1: Product recommendation
product = st.text_input(
    "Enter Product Name"
)
if st.button("Get Recommendations"):
for item in recommendations:
    st.success(item)

Module 2: Customer segmentaion

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
