import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Page settings
st.set_page_config(page_title="Accident Detection", page_icon="🚦")

# CSS styling
page_bg = """
<style>
.stApp {
    background-color: black;
    color: white;
}

h1 {
    text-align: center;
    color: white;
}

p {
    color: white;
}

/* Fix uploader text */
[data-testid="stFileUploader"] label {
    color: white;
}

/* Fix browse button */
button[kind="secondary"] {
    background-color: white;
    color: black;
}

/* Style detect button */
.stButton>button {
    background-color: red;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 200px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.title("🚦📹 Real-Time Accident Detection from CCTV Images")

st.write("Upload a CCTV road image to detect whether an accident has occurred.")

# Load model
model = load_model("accident_model02 (1).h5")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img,(128,128))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    if st.button("Detect Accident"):

        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            st.error("🚨 Accident Detected 🚨")
        else:
            st.success("✅ No Accident Detected")

# Footer
st.markdown("---")
st.markdown(" Accident Detection using CNN")