import os
import sys
import unittest
import cv2
import numpy as np

print(os.getcwd())

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.image_processing.image_to_css import image_to_css

class TestImageProcessing(unittest.TestCase):
    def test_image_to_css(self):
        # Define the path to a test image
        test_image_path = 'tests/mockup/Signup.png'

        # Check if the image file exists
        self.assertTrue(os.path.exists(test_image_path), "Image file does not exist")

        # Try to read the image
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image, "Failed to read image")

        # Check the depth of the image
        print("Image depth:", image.dtype)

        # Check the number of channels in the image
        print("Image channels:", image.shape)

        # Convert the image to 8-bit
        image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert the image to RGB using a different method
        image = np.stack((image,) * 3, axis=-1)

        # Call the image_to_css function with the test image
        css = image_to_css(test_image_path)

        # Assert that the returned CSS is a string
        self.assertIsInstance(css, str)

        # Assert that the CSS contains some expected values
        # This will depend on the specific test image and the expected output
        self.assertIn('.button {', css)
        self.assertIn('.text_field {', css)
        self.assertIn('.image {', css)

if __name__ == '__main__':
    unittest.main()
