import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# 1. Load Labels
@st.cache_data
def load_labels():
    with open('config/class_indices.json', 'r') as f:
        return json.load(f)

labels = load_labels()
num_classes = len(labels)

# 2. Build the EXACT Model from your Notebook
@st.cache_resource
def load_model_from_weights():
    # A. Base VGG16 (Must match your notebook settings)
    base_model = tf.keras.applications.VGG16(
        include_top=False, 
        weights=None,  # We don't need ImageNet weights, we have yours
        input_shape=(64, 64, 3)
    )
    
    # B. Rebuild the exact layers you defined in Task 3
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # You used this, not Flatten!
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # C. Load your trained weights
    # compile=False prevents the optimizer crash
    model.load_weights('models/fruit_classifier.keras')
    
    return model

try:
    model = load_model_from_weights()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. App Interface
st.title("🍎 Fruit Classifier AI")
st.write("Upload an image of a fruit to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess (Must match your Train Generator: 1/255 scale)
    img = image.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Your notebook used rescale=1.0/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if st.button('Classify'):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_label = labels.get(str(predicted_class_index), "Unknown")
        confidence = np.max(prediction) * 100
        
        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")