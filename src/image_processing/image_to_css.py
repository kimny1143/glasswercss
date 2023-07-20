import cv2
from skimage import io, color
import numpy as np
import tensorflow as tf


# Load the pre-trained model
model = tf.saved_model.load('path_to_pretrained_model')


def preprocess_image(image, size=(800, 800)):
    # Resize the image to the desired size
    image = cv2.resize(image, size)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0

    # Add an extra dimension to the image
    image = np.expand_dims(image, axis=2)

    return image

def detect_elements(image):
    # Convert the image to the format expected by the model
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

    # Use the model to detect elements in the image
    result = model(image)

    # Extract the bounding boxes and labels from the result
    bounding_boxes = result['detection_boxes']
    labels = result['detection_classes']

    # Convert the labels to a more human-readable format
    # This will depend on the specific model and dataset used
    labels = convert_labels(labels)

    # Return a list of detected elements, each represented as a bounding box and a label
    elements = list(zip(bounding_boxes, labels))

    return elements

def convert_labels(labels):
    # Define a mapping from integers to element names
    # This will depend on the specific model and dataset used
    label_mapping = {
        1: 'button',
        2: 'text field',
        3: 'image',
        # Add more mappings as needed
    }

    # Convert the labels to element names using the mapping
    labels = [label_mapping[label] for label in labels]

    return labels



def classify_elements(image, elements):
    element_types = []

    for element in elements:
        # Extract the bounding box and label from the element
        bounding_box, label = element

        # Extract the corresponding part of the image
        x1, y1, x2, y2 = bounding_box
        element_image = image[y1:y2, x1:x2]

        # Preprocess the element image
        element_image = preprocess_element_image(element_image)

        # Use the model to classify the type of the element
        element_type = model.predict(element_image)

        # Append the element type to the list
        element_types.append(element_type)

    return element_types

import cv2
import numpy as np

def preprocess_element_image(element_image, size=(64, 64)):
    # Resize the image to the desired size
    element_image = cv2.resize(element_image, size)

    # If the image is not grayscale, convert it to grayscale
    if element_image.shape[2] != 1:
        element_image = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)

    # Normalize the pixel values to the range [0, 1]
    element_image = element_image / 255.0

    # Add an extra dimension to the image
    element_image = np.expand_dims(element_image, axis=2)

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

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Detect elements in the image
    elements = detect_elements(preprocessed_image)

    # Classify the type of each element
    element_types = classify_elements(preprocessed_image, elements)

    # Generate the corresponding CSS
    css = generate_css(elements, element_types)

    return css
