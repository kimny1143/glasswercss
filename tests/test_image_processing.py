import unittest
from ..src.image_processing.image_to_css import image_to_css

class TestImageProcessing(unittest.TestCase):
    def test_image_to_css(self):
        # Define the path to a test image
        test_image_path = 'path_to_test_image'

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
