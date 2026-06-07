import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# ---------------------------------------------------

# Page Configuration

# ---------------------------------------------------

st.set_page_config(
page_title="Cat vs Dog Classifier",
page_icon="🐱",
layout="wide"
)

# ---------------------------------------------------

# Custom CSS

# ---------------------------------------------------

st.markdown("""

<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #4FC3F7;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #B0BEC5;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #1E1E1E;
    border: 2px solid #4FC3F7;
}

.prediction {
    font-size: 40px;
    font-weight: bold;
    color: #00E676;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 16px;
}

</style>

""", unsafe_allow_html=True)

# ---------------------------------------------------

# Load Deep Learning Model

# ---------------------------------------------------


@st.cache_resource
def load_model_file():

    load_model(r"model/model_vgg.keras")

    return model


model = load_model_file()

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------

def predict_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    probability = float(prediction[0][0])

    if probability > 0.5:

        label = "🐶 Dog"

        confidence = probability

    else:

        label = "🐱 Cat"

        confidence = 1 - probability

    return label, confidence

# ---------------------------------------------------

# Header Section

# ---------------------------------------------------

st.markdown(
'<div class="title">🐱🐶 Cat vs Dog Classification System</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Deep Learning Powered Image Classification using CNN</div>',
unsafe_allow_html=True
)

st.write("")

# ---------------------------------------------------

# Upload Section

# ---------------------------------------------------

uploaded_file = st.file_uploader(
"📤 Upload Cat or Dog Image",
type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # RIGHT COLUMN
    with col2:

        st.subheader("Prediction Result")

        with st.spinner("Analyzing Image..."):

            label, confidence = predict_image(image)

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction">{label}</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.progress(int(confidence * 100))

        st.metric(
            "Confidence Score",
            f"{confidence:.2%}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

else:

    st.info(
        "Please upload a Cat or Dog image."
    )

# ---------------------------------------------------

# Footer

# ---------------------------------------------------

st.markdown(
'<div class="footer">Developed using Streamlit | Keras | TensorFlow | Deep Learning</div>',
unsafe_allow_html=True
)
