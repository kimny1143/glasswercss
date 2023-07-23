import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import torchvision.models as models

def load_classification_model():
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=True)

    # Set the model to evaluation mode
    model.eval()

    return model

def detect_elements(image):
    # Check if the image is grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Convert the image to the format expected by the model
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = F.to_tensor(image)  # Convert the image to a PyTorch tensor
    image = image.unsqueeze(0)  # Add an extra dimension for the batch

    # Use the model to detect elements in the image
    with torch.no_grad():
        result = model(image)

    # Extract the bounding boxes and labels from the result
    bounding_boxes = result[0]['boxes']
    labels = result[0]['labels']

    # Convert the labels to a more human-readable format
    labels = convert_labels(labels)

    # Return a list of detected elements, each represented as a bounding box and a label
    elements = list(zip(bounding_boxes, labels))

    return elements

def preprocess_image(image, size=(800, 800)):
    # Resize the image to the desired size
    image = cv2.resize(image, size)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0

    # Convert the image back to uint8
    image = (image * 255).astype(np.uint8)

    # Add an extra dimension to the image
    image = np.expand_dims(image, axis=2)

    return image

def convert_labels(labels):
    label_mapping = {
        0: 'background',
        1: 'text',
        2: 'title',
        3: 'list',
        4: 'table',
        5: 'figure',
        6: 'button',  # ボタンのラベルを追加
        7: 'text field',  # テキストフィールドのラベルを追加
        8: 'image',  # 画像のラベルを追加
    }
    labels = [label_mapping.get(label, 'unknown') for label in labels]
    return labels

def classify_elements(image, elements, classification_model):
    element_types = []

    for element in elements:
        # Extract the bounding box and label from the element
        bounding_box, label = element

        # Convert the bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, bounding_box)

        # Extract the corresponding part of the image
        element_image = image[y1:y2, x1:x2]

        # Preprocess the element image
        element_image = preprocess_element_image(element_image)

        # Convert the image to a PyTorch tensor
        element_image = torch.from_numpy(element_image).permute(2, 0, 1).float().unsqueeze(0)

        # Use the model to classify the type of the element
        output = classification_model(element_image)

        # Get the probabilities and find the class with the highest probability
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        element_type = probabilities.argmax().item()

        # Add the element type to the list
        element_types.append(element_type)

    return element_types

def preprocess_element_image(element_image, size=(64, 64)):
    # Resize the image to the desired size
    element_image = cv2.resize(element_image, size)

    # If the image is not grayscale, convert it to grayscale
    if len(element_image.shape) == 3 and element_image.shape[2] != 1:
        element_image = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)

    # Normalize the pixel values to the range [0, 1]
    element_image = element_image / 255.0

    # Convert the image back to 8-bit depth
    element_image = (element_image * 255).astype(np.uint8)

    # Convert the image to RGB
    element_image = cv2.cvtColor(element_image, cv2.COLOR_GRAY2RGB)

    # Normalize the pixel values to the range [0, 1] again
    element_image = element_image / 255.0

    return element_image

def generate_css(elements, element_types):
    css = ""

    for element, element_type in zip(elements, element_types):
        # Extract the bounding box and label from the element
        bounding_box, label = element

        # Generate the CSS for this element
        # This will depend on the type of the element and its properties
        if element_type == 'button':
            css += generate_button_css(bounding_box, label)
        elif element_type == 'text field':
            css += generate_text_field_css(bounding_box, label)
        elif element_type == 'image':
            css += generate_image_css(bounding_box, label)
        # Add more cases as needed

    return css

def generate_button_css(bounding_box, label):
    # Extract the coordinates of the bounding box
    x1, y1, x2, y2 = bounding_box

    # Calculate the width and height of the button
    width = x2 - x1
    height = y2 - y1

    # Generate the CSS for the button
    css = f"""
    .{label} {{
        position: absolute;
        left: {x1}px;
        top: {y1}px;
        width: {width}px;
        height: {height}px;
        background-color: #007BFF;
        border: none;
        color: white;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }}
    """

    return css

def generate_text_field_css(bounding_box, label):
    # Extract the coordinates of the bounding box
    x1, y1, x2, y2 = bounding_box

    # Calculate the width and height of the text field
    width = x2 - x1
    height = y2 - y1

    # Generate the CSS for the text field
    css = f"""
    .{label} {{
        position: absolute;
        left: {x1}px;
        top: {y1}px;
        width: {width}px;
        height: {height}px;
        padding: 12px 20px;
        box-sizing: border-box;
        border: 2px solid #ccc;
        border-radius: 4px;
        background-color: white;
        font-size: 16px;
        resize: none;
    }}
    """

    return css

def generate_image_css(bounding_box, label):
    # Extract the coordinates of the bounding box
    x1, y1, x2, y2 = bounding_box

    # Calculate the width and height of the image
    width = x2 - x1
    height = y2 - y1

    # Generate the CSS for the image
    css = f"""
    .{label} {{
        position: absolute;
        left: {x1}px;
        top: {y1}px;
        width: {width}px;
        height: {height}px;
    }}
    """

    return css

def image_to_css(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path {image_path}")

    # Detect elements in the image
    elements = detect_elements(image)
    print(f"Detected elements: {elements}")  # Debugging print statement

    if not elements:
        raise ValueError("No elements detected in the image")

    # Load the classification model
    classification_model = load_classification_model()

    # Classify the type of each element
    element_types = classify_elements(image, elements, classification_model)
    print(f"Element types: {element_types}")  # Debugging print statement

    # Generate the corresponding CSS
    css = generate_css(elements, element_types)

    return css
