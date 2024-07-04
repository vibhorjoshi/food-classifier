import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image

# Custom load function to handle the 'groups' parameter issue
def custom_load_function(path):
    def remove_groups(config):
        if 'groups' in config:
            del config['groups']
        return config
    
    with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': lambda **kwargs: tf.keras.layers.DepthwiseConv2D(**remove_groups(kwargs))}):
        model = tf.keras.models.load_model(path)
    return model

# Load the pre-trained model using the custom function
model_path = 'final_food_classifier.h5'
model = custom_load_function(model_path)

# Reading class names
df = pd.read_csv('class_names.csv')
label = df['Label']

# Streamlit app
st.title("Food Classification App")

# Upload image through streamlit
uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "jpeg"])

# Function to preprocess the image
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction on the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Read the uploaded image using pillow and resize
    img = Image.open(uploaded_file).resize((224, 224))
    
    # Preprocess and predict
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    
    # Display the top prediction
    st.subheader("Prediction:")
    predicted_class = label[np.argmax(predictions)]
    st.write(predicted_class)
    
    # Optionally, display the top 3 predictions with their probabilities
    top_3 = np.argsort(predictions[0])[-3:][::-1]
    st.subheader("Top 3 Predictions:")
    for i in top_3:
        st.write(f"{label[i]}: {predictions[0][i]*100:.2f}%")