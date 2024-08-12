import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage import color, filters
from skimage.filters import threshold_otsu

# Load image
img = cv2.imread('flower.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the original image as Figure 1
plt.figure(1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')  # Hide axis
plt.show()

# Global Thresholding using Otsu's Method
_, bw = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(2)
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(bw, cmap='gray'), plt.title('Binary Image'), plt.axis('off')
plt.show()

# Multi-level Thresholding using Otsu's Method
thresholds = filters.threshold_multiotsu(gray_img, classes=3)
regions = np.digitize(gray_img, bins=thresholds)

# Convert segmented image to RGB
seg_img_rgb = color.label2rgb(regions, image=img, bg_label=0)

plt.figure(3)
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(seg_img_rgb), plt.title('Segmented Image'), plt.axis('off')
plt.show()

# Histogram Thresholding using Otsu's Method
counts, bins = np.histogram(gray_img.flatten(), bins=16, range=(0, 256))
otsu_thresh = threshold_otsu(gray_img)
_, bw_otsu = cv2.threshold(gray_img, otsu_thresh, 255, cv2.THRESH_BINARY)

plt.figure(4)
plt.bar(bins[:-1], counts, width=bins[1] - bins[0])
plt.title('16-bin Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Count')
plt.show()

# Compute histogram counts
counts, bins = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))

# Compute global threshold using Otsu's method
otsu_thresh = filters.threshold_otsu(gray_img)

# Create a binary image using the computed threshold
_, bw = cv2.threshold(gray_img, otsu_thresh, 255, cv2.THRESH_BINARY)

# Convert binary image to RGB for display
bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

# Display the original image and the binary image side by side
plt.figure(5)
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(bw_rgb), plt.title('Binary Image'), plt.axis('off')
plt.show()

# Region-Based Segmentation using K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(gray_img.reshape(-1, 1))
seg_img = kmeans.labels_.reshape(gray_img.shape)
seg_img_rgb = color.label2rgb(seg_img, image=img, bg_label=0)

plt.figure(6)
plt.imshow(seg_img_rgb)
plt.title('Labeled Image')
plt.axis('off')
plt.show()

# Connected-Component Labeling
_, bin_img = cv2.threshold(gray_img, otsu_thresh, 255, cv2.THRESH_BINARY)
num_labels, labeled_img = cv2.connectedComponents(bin_img)
colored_labels = cv2.applyColorMap(np.uint8(labeled_img * 255 / num_labels), cv2.COLORMAP_JET)

print('Number of connected components:', num_labels)

plt.figure(7)
plt.imshow(colored_labels)
plt.title('Labeled Image')
plt.axis('off')
plt.show()

# Parameter Modifications with Noise
# Add noise to the RGB image
noise_img = np.clip(img + np.random.normal(0, 25, img.shape).astype(np.uint8), 0, 255)

# Segment the noisy RGB image
kmeans_noise = KMeans(n_clusters=3, random_state=0).fit(noise_img.reshape(-1, 3))
labels_noise = kmeans_noise.labels_.reshape(noise_img.shape[:2])
label_overlay_noise = color.label2rgb(labels_noise, image=noise_img, bg_label=0)

plt.figure(8)
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)), plt.title('Noisy Image'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(label_overlay_noise), plt.title('Segmented Image with Noise'), plt.axis('off')
plt.show()

# Segment the original RGB image into regions using K-means
kmeans_rgb = KMeans(n_clusters=2, random_state=0).fit(img.reshape(-1, 3))
labels_rgb = kmeans_rgb.labels_.reshape(img.shape[:2])
label_overlay_rgb = color.label2rgb(labels_rgb, image=img, bg_label=0)

plt.figure(9)
plt.imshow(label_overlay_rgb)
plt.title('Segmented Image')
plt.axis('off')
plt.show()

# Create and apply Gabor filters
def gabor_filter(img, wavelength, orientation):
    filters = []
    for theta in orientation:
        theta = np.deg2rad(theta)
        for lambda_ in wavelength:
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, lambda_, 0.5, 0, cv2.CV_32F)
            filters.append(kernel)
    return filters

wavelength = [2 ** i * 3 for i in range(6)]
orientation = list(range(0, 180, 45))
gabor_kernels = gabor_filter(gray_img, wavelength, orientation)

gabor_mag = np.zeros_like(gray_img, dtype=np.float32)
for kernel in gabor_kernels:
    filtered_img = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
    gabor_mag = np.maximum(gabor_mag, np.abs(filtered_img))

plt.figure(10)
num_kernels = len(gabor_kernels)
for i in range(num_kernels):
    plt.subplot(4, 6, i + 1)
    plt.imshow(cv2.filter2D(gray_img, cv2.CV_32F, gabor_kernels[i]), cmap='gray')
    plt.axis('off')  # Hide axis
plt.show()

# Smooth each filtered image
for i, kernel in enumerate(gabor_kernels):
    sigma = 0.5 * wavelength[i % len(wavelength)]
    gabor_mag = cv2.GaussianBlur(gabor_mag, (0, 0), sigma)

plt.figure(11)
for i in range(num_kernels):
    plt.subplot(4, 6, i + 1)
    plt.imshow(gabor_mag, cmap='gray')
    plt.axis('off')  # Hide axis
plt.show()

# Feature set and K-means clustering
x, y = np.meshgrid(np.arange(gray_img.shape[1]), np.arange(gray_img.shape[0]))
feature_set = np.stack([gray_img, gabor_mag, x, y], axis=-1)

feature_set_reshaped = feature_set.reshape(-1, feature_set.shape[-1])
kmeans_feature = KMeans(n_clusters=2, random_state=0).fit(feature_set_reshaped)
labels_feature = kmeans_feature.labels_.reshape(gray_img.shape)
label_overlay_feature = color.label2rgb(labels_feature, image=img, bg_label=0)

plt.figure(12)
plt.imshow(label_overlay_feature)
plt.title('Labeled Image with Additional Pixel Information')
plt.axis('off')
plt.show()
