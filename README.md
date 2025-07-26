# FIC
This project is about building a machine learning model to recognize and classify images of Indian 
food into different categories (like "burger," "butter naan," "chai," etc.). It uses a pre-trained 
ResNet18 deep learning model to classify images based on their visual features. 
1. What Does the Project Do? 
• Input: Images of Indian food. 
• Output: The predicted category of the food item (e.g., "burger," "idli," "samosa"). 
Step-by-Step Explanation 
1. Preparing the Data 
• The dataset contains images of various Indian food items, organized into folders. Each folder 
corresponds to a food category. 
• The project divides the dataset into: 
o Training Data (80%): Used to teach the model. 
o Validation Data (20%): Used to check how well the model performs on unseen 
images. 
2. Transforming the Images 
• Images are preprocessed to ensure they are suitable for the model: 
o Resized to a fixed size of 224x224 pixels (as required by the ResNet18 model). 
o Augmented with random flips, rotations, and color changes to make the model more 
robust. 
o Converted to tensors and normalized to align with the pre-trained model's 
expectations. 
3. Loading the Model 
• ResNet18, a popular deep learning model, is loaded with pre-trained weights from ImageNet 
(a massive dataset for general image recognition). 
• The final layer of ResNet18 is modified to match the number of food categories in the 
dataset. 
4. Training the Model 
• The model is trained using: 
o Training Data: Images and their corresponding labels. 
o Loss Function: Measures the difference between the model's predictions and the 
actual labels. 
o Optimizer: Adjusts the model’s parameters to minimize the loss function. 
• Process: 
o The model sees batches of images, predicts their categories, calculates the loss, and 
updates its parameters. 
o After training on the entire dataset (one epoch), the model is evaluated on the 
validation data. 
• Goal: To maximize accuracy on the validation dataset by finding the best model. 
5. Evaluating the Model 
• After training, the model is tested using validation data. Metrics like accuracy, precision, 
recall, and confusion matrix help assess its performance. 
6. Making Predictions 
• For a new food image: 
o The image is preprocessed (resized, normalized, etc.). 
o The model predicts its category. 
o The predicted category is displayed alongside the image. 
Input and Output 
1. Input: 
o A folder of labeled food images (training and validation data). 
o A single image for prediction (during inference). 
2. Output: 
o During training: Validation accuracy and the saved model with the best performance. 
o During prediction: The category (label) of the input food image. 
Key Features 
1. Training Results: 
o The model achieves ~73% validation accuracy, meaning it correctly classifies about 
73% of unseen food images. 
2. Prediction Example: 
o An image of a burger is fed into the model, and it predicts the category as "burger." 
3. Detailed Performance Metrics: 
o A classification report shows how well the model performs for each food category 
(precision, recall, F1-score). 
o A confusion matrix visualizes where the model gets confused between categories. 
Final Deliverables 
1. Trained Model: A file (best_model.pth) that contains the best-performing version of the 
model. 
2. Evaluation Tools: Scripts to generate classification reports and confusion matrices. 
3. Prediction Functionality: A function to classify new food images
