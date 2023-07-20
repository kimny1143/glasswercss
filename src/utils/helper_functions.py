import cv2

def read_image(image_path):
    # Use cv2.imread to read the image from the file
    image = cv2.imread(image_path)

    # cv2.imread returns None if it can't read the image file for any reason
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    return image
