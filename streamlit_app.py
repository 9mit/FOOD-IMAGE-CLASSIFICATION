# streamlit_app.py
import streamlit as st
from app.chatbot import generate_response
from app.scraper import get_recipe
import os
import random
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


# Input from user
st.title("Chat with the Food Bot")

# Input from user
user_input = st.text_input("Ask me anything about food:")

# Frontend code
if user_input:
    # Check if the user is asking for a recipe
    if "recipe" in user_input.lower():
        # Extract the food item from the user input
        food_item = user_input.lower().replace("recipe", "").strip()
        if food_item:
            st.write(f"Fetching recipe for: {food_item.capitalize()}")
            try:
                # Call the get_recipe function
                recipe = get_recipe(food_item)
                st.write(recipe)
            except Exception as e:
                st.error(f"Could not fetch recipe: {e}")
        else:
            st.warning("Please specify a food item for the recipe.")
    else:
        # Otherwise, use the chatbot to generate a response
        response = generate_response(user_input)
        st.write(response)

# Example usage to test the function
food_item = random.choice(["burger", "butter_naan", "chai", "chapati", "chole bhature", "dal makhani", "dhokla", "fried_rice", "idli", "jalebi", "kaathi rolls", "kadai paneer", "kulfi", "masala dosa", "momos", "paani puri", "pakode", "pav bhaji", "pizza", "samosa"])
print(get_recipe(food_item))


# Evaluate the model
if st.button("Evaluate Model"):
    all_labels, all_preds, report, matrix = evaluate_model()
    st.write("Classification Report:")
    report_dict = classification_report(all_labels, all_preds, output_dict=True, target_names=train_dataset.classes)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)
    st.write("Confusion Matrix:")
    st.write(matrix)
