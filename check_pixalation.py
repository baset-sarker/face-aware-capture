import cv2
import numpy as np

def check_pixelation(image_path, threshold=50):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the horizontal and vertical gradients using Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the gradients
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Calculate the mean gradient magnitude
    mean_gradient = np.mean(gradient_magnitude)

    # Determine if the image is pixelated based on the mean gradient magnitude
    if mean_gradient < threshold:
        print("The image is pixelated.")
    else:
        print("The image is not pixelated.")

# Example usage
image_path = 'path_to_your_image.jpg'
check_pixelation(image_path)



import cv2
import numpy as np

def check_washed_out(image_path, threshold=0.1):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the standard deviation of pixel intensities
    std_dev = np.std(gray)

    # Calculate the mean pixel intensity
    mean_intensity = np.mean(gray)

    # Calculate the coefficient of variation
    if mean_intensity == 0:
        coefficient_of_variation = 0
    else:
        coefficient_of_variation = std_dev / mean_intensity

    # Determine if the image is washed out based on the coefficient of variation
    if coefficient_of_variation < threshold:
        print("The image is washed out.")
    else:
        print("The image is not washed out.")

# Example usage
image_path = 'path_to_your_image.jpg'
check_washed_out(image_path)
