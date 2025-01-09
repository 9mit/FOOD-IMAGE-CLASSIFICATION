# streamlit_app.py
import streamlit as st
import os
from PIL import Image
from app.ml_model import predict_image, evaluate_model, train_dataset
import pandas as pd
from sklearn.metrics import classification_report

# Set up the Streamlit app
st.title("Food Image Classification App")

# Include custom CSS
st.markdown(
    """
    <style>
    @import url('static/css/style.css');
    </style>
    """,
    unsafe_allow_html=True
)

# Include custom JavaScript
st.markdown(
    """
    <script src="static/js/script.js"></script>
    """,
    unsafe_allow_html=True
)

# Upload image for prediction
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file
    image_path = os.path.join('data/uploads', uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the image
    predicted_class, report, matrix, nutrition = predict_image(image_path)

    # Display the predicted class
    st.write(f"Predicted Class: {predicted_class}")

    # Display the image
    image = Image.open(image_path)
    st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_container_width=True)

    # Display the classification report
    st.write("Classification Report:")
    report_dict = classification_report([0], [0], output_dict=True, target_names=[predicted_class])
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    # Display the confusion matrix
    st.write("Confusion Matrix:")
    st.write(matrix)

    # Display the nutritional information
    st.write("Nutritional Information:")
    st.write(nutrition)

# Evaluate the model
if st.button("Evaluate Model"):
    all_labels, all_preds, report, matrix = evaluate_model()
    st.write("Classification Report:")
    report_dict = classification_report(all_labels, all_preds, output_dict=True, target_names=train_dataset.classes)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)
    st.write("Confusion Matrix:")
    st.write(matrix)
