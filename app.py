import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="cnn_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define function to make predictions
def predict(image: Image.Image):
    img = image.resize((224, 224))  # Resize to the model's expected size
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set tensor to the input image
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data, axis=1)

    return prediction

# Streamlit UI
st.title("Gabah Classification Prediction")
st.write("Upload a rice image to predict its variety.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write("Classifying...")
        prediction = predict(image)
        labels = ["Basmati", "IR64", "Pandan Wangi"]
        st.write(f"Prediction: {labels[prediction[0]]}")

