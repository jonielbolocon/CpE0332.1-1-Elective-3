import cv2
import numpy as np
import matplotlib.pyplot as plt

#Figure 1: Acquire an Image of a Flower
# Read an image
img = cv2.imread('flower.jpg')

# Display the original image
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')
plt.show()

#Figure 2: Grayscale, Contrast Enhancement, and its Histogram
# Convert to grayscale if the image is RGB
if len(img.shape) == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img

# Display the grayscale image
plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')
plt.show()

# Contrast enhancement 
img_contrast_enhanced = cv2.normalize(img_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Display the contrast-enhanced image
plt.figure()
plt.imshow(img_contrast_enhanced, cmap='gray')
plt.axis('off')
plt.title('Contrast Enhanced Image')
plt.show()

#Figure 3: Histogram Equalized and Average Filtered Image and Its Histogram
# Histogram equalization
img_histeq = cv2.equalizeHist(img_gray)

# Display the histogram equalized image
plt.figure()
plt.imshow(img_histeq, cmap='gray')
plt.axis('off')
plt.title('Equalized Image')
plt.show()

# Filtering using average filter
h_avg = np.ones((5, 5), np.float32) / 25
img_avg_filtered = cv2.filter2D(img_gray, -1, h_avg)

# Display the average filtered image
plt.figure()
plt.imshow(img_avg_filtered, cmap='gray')
plt.axis('off')
plt.title('Filtered Image (Average)')
plt.show()

#Figure 4: Median Filtered Image and Its Histogram
# Filtering using median filter
img_median_filtered = cv2.medianBlur(img_gray, 5)

# Display the median filtered image
plt.figure()
plt.imshow(img_median_filtered, cmap='gray')
plt.axis('off')
plt.title('Filtered Image (Median)')
plt.show()

# Display histograms for comparison

# Grayscale histogram
plt.figure()
plt.hist(img_gray.ravel(), 256, [0, 256])
plt.title('Histogram of Grayscale')
plt.show()

# Enhanced histogram (contrast adjustment)
plt.figure()
plt.hist(img_contrast_enhanced.ravel(), 256, [0, 256])
plt.title('Histogram of Enhanced Image')
plt.show()

# Equalized histogram
plt.figure()
plt.hist(img_histeq.ravel(), 256, [0, 256])
plt.title('Histogram of Equalized Image')
plt.show()

# Histogram (Average Filtered)
plt.figure()
plt.hist(img_avg_filtered.ravel(), 256, [0, 256])
plt.title('Histogram of Average Filtered Image')
plt.show()

# Histogram (Median Filtered)
plt.figure()
plt.hist(img_median_filtered.ravel(), 256, [0, 256])
plt.title('Histogram of Median Filtered Image')
plt.show()