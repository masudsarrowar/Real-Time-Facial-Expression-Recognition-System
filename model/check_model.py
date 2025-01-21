import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import logging
import os
import time  # Import the time module for sleep functionality
from emotion_model import EmotionCNN  # Import the custom CNN model for emotion classification

# Configure logging to capture important information during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the custom CNN model
model = EmotionCNN()

# Load the pre-trained model weights (ensure the correct path to the model file)
state_dict = torch.load('./emotion_model.pth', map_location=torch.device('cpu'), weights_only=True)  # Safely load model weights
model.load_state_dict(state_dict)  # Load the state dictionary into the model
model.eval()  # Set the model to evaluation mode, ensuring layers like dropout are disabled

# Define the image preprocessing pipeline for consistent input format to the model
preprocess = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize input image to 48x48 pixels, which is the model's expected input size
    transforms.ToTensor(),        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the image tensor (adjust based on model training configuration)
])

def preprocess_image(img_path):
    """
    Preprocess the input image to match the format expected by the model.
    Args:
        img_path (str): The file path to the image to be processed.
    Returns:
        torch.Tensor: A tensor representation of the processed image, with a batch dimension.
    """
    img = Image.open(img_path).convert('L')  # Open the image and convert it to grayscale (assuming the model is trained on grayscale images)
    img_tensor = preprocess(img)  # Apply the preprocessing transformations
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension (model expects input shape: [batch_size, channels, height, width])
    return img_tensor

def predict_image(img_path):
    """
    Predict the emotion class for a given image using the pre-trained model.
    Args:
        img_path (str): The file path to the image to be predicted.
    Returns:
        int: The index of the predicted emotion class.
    """
    img_tensor = preprocess_image(img_path)  # Preprocess the image for model input
    with torch.no_grad():  # Disable gradient calculations during inference to reduce memory usage
        outputs = model(img_tensor)  # Perform a forward pass to get model outputs
    _, predicted_class = torch.max(outputs, 1)  # Extract the index of the class with the highest probability
    return predicted_class.item()  # Return the predicted class index as an integer

def display_image(img_path):
    """
    Display the input image for visual inspection.
    Args:
        img_path (str): The file path to the image to be displayed.
    """
    img = Image.open(img_path)  # Open the image file
    plt.imshow(img, cmap='gray')  # Display the image in grayscale
    plt.axis('off')  # Hide axes for a cleaner display
    plt.show()  # Show the image

if __name__ == "__main__":
    logging.info("Starting the periodic prediction process...")  # Log the initiation of the process

    # Define the path to the input image
    img_path = '../images/latest_image.jpg'

    while True:  # Infinite loop to repeat the process every 30 seconds
        # Perform the class prediction for the image
        predicted_class = predict_image(img_path)

        # Class labels corresponding to the model's output (adjust to match your model's class names)
        class_names = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust']
        predicted_class_name = class_names[predicted_class]  # Retrieve the predicted class label

        # Log the predicted emotion class
        logging.info(f"Predicted emotion class for the image: {predicted_class_name}")

        # Display the image for visual confirmation
        display_image(img_path)

        # Sleep for 30 seconds before making the next prediction
        time.sleep(30)  # Delay the next prediction for 30 seconds
