import cv2
import numpy as np

orig = cv2.imread("input.bmp")
encoded = cv2.imread("encoded_output.bmp")

diff = np.sum(orig != encoded)
print(f"Number of pixels changed: {diff}")
