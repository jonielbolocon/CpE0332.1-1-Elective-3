#Bolocon, Joniel R.
import cv2
import numpy as np

# Read the image
img = cv2.imread('flower.jpg')

# Display the image
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Get image dimensions (rows, columns, color channels)
rows, cols, channels = img.shape
print('Image size:', rows, 'x', cols, 'x', channels)

# Check color model (grayscale or RGB)
if channels == 1:
    print('Color Model: Grayscale')
else:
    print('Color Model: RGB')

# Access individual pixels (example: center pixel)
center_row = rows // 2
center_col = cols // 2
center_pixel = img[center_row, center_col]
print('Center pixel value:', center_pixel)

# Basic arithmetic operations (add constant value to all pixels)
brightened_img = np.clip(img, 75, 245).astype(np.uint8)
cv2.imshow('Brightened Image', brightened_img)
cv2.waitKey(0)

# Basic geometric operation (flipping image horizontally)
flipped_img = cv2.flip(img, 1)
cv2.imshow('Horizontally Flipped Image', flipped_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
