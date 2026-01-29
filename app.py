import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# 1. Load the Model and Labels (Cached for performance)
@st.cache_resource
def load_model():
    # Update path to look inside the 'models' folder
    return tf.keras.models.load_model('models/fruit_classifier_model.h5')

@st.cache_data
@st.cache_data
def load_labels():
    # Update path to look inside the 'config' folder
    with open('config/class_indices.json', 'r') as f:
        return json.load(f)

model = load_model()
labels = load_labels()

# 2. App UI
st.title("🍎 Fruit Classifier AI")
st.write("Upload an image of a fruit, and this VGG16-powered AI will identify it.")

uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 3. Preprocess the image (MUST MATCH YOUR NOTEBOOK)
    # Resize to 64x64 as defined in your train_generator
    img = image.resize((64, 64))
    # Convert to array
    img_array = np.array(img)
    # Rescale by 1/255 as done in your ImageDataGenerator
    img_array = img_array / 255.0
    # Add batch dimension (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Make Prediction
    if st.button('Classify'):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_label = labels[str(predicted_class_index)]
        confidence = np.max(prediction) * 100

        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")