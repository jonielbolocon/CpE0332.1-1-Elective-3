import cv2
#Bolocon, Joniel R.

# Read the image
img = cv2.imread('flower.jpg')

# Rotate the image by 30 degrees
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)  # 30 degrees, scale = 1.0
rotate_img = cv2.warpAffine(img, rotation_matrix, (w, h))

# Flip the rotated image horizontally
flipped_horizontal = cv2.flip(rotate_img, 1)  # 1 for horizontal flip

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Rotated by 30 Degrees', rotate_img)
cv2.imshow('Flipped Horizontally', flipped_horizontal)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()