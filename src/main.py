from image_processing.image_to_css import image_to_css
from utils.helper_functions import read_image

def main(image_path):
    image = read_image(image_path)
    css = image_to_css(image)
    print(css)

if __name__ == "__main__":
    image_path = "../tests/mockup/Signup.png"
    css = image_to_css(image_path)
    print(css)
