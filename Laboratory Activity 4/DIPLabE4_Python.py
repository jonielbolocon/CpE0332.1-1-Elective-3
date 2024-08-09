import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('flower.jpg')  # Replace with the path to your image file
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure()
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Convert to grayscale if the image is RGB
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Add motion blur to the image
len = 21
psf = np.zeros((len, len))
center = len // 2
for i in range(len):
    psf[i, center] = 1
psf = psf / psf.sum()
img_blur = ndimage.convolve(img_gray.astype(float), psf, mode='wrap')

# Display the blurred image
plt.figure()
plt.imshow(img_blur, cmap='gray')
plt.title('Motion Blurred Image')
plt.axis('off')

# Gaussian filtering
h_gaussian = cv2.getGaussianKernel(5, 1)
h_gaussian = h_gaussian @ h_gaussian.T
img_gaussian_filtered = cv2.filter2D(img_blur, -1, h_gaussian)

# Display the Gaussian filtered image
plt.figure()
plt.imshow(img_gaussian_filtered, cmap='gray')
plt.title('Filtered Image (Gaussian)')
plt.axis('off')

# Edge Detection
img_edges = cv2.Canny(img_gray, 25, 75)

# Display the edge-detected image
plt.figure()
plt.imshow(img_edges, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

# Sharpening using unsharp masking
blurred = cv2.GaussianBlur(img_blur, (5, 5), 1.0)
img_sharpened = cv2.addWeighted(img_blur, 1.5, blurred, -0.5, 0)

# Display the sharpened image
plt.figure()
plt.imshow(img_sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

# Add Gaussian noise and remove it using median filter
img_noisy = np.clip(img_gray + np.random.normal(0, 20, img_gray.shape), 0, 255).astype(np.uint8)
img_noisy_removed = cv2.medianBlur(img_noisy, 5)

# Display the noisy image
plt.figure()
plt.imshow(img_noisy, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

# Display the noise-removed image
plt.figure()
plt.imshow(img_noisy_removed, cmap='gray')
plt.title('Noise Removed Image')
plt.axis('off')

# Deblurring
def wiener_deconvolution(img, kernel, noise_var):
    # Compute the Fourier transform of the image and kernel
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel, s=img.shape)
    
    # Compute Wiener filter
    kernel_fft_conj = np.conj(kernel_fft)
    wiener_filter = kernel_fft_conj / (np.abs(kernel_fft) ** 2 + noise_var)
    
    # Apply Wiener filter in the frequency domain
    deblurred_fft = img_fft * wiener_filter
    
    # Convert back to spatial domain
    deblurred_img = np.abs(np.fft.ifft2(deblurred_fft))
    
    return np.uint8(np.clip(deblurred_img, 0, 255))

# Define the kernel (PSF) for deblurring
kernel_size = 21
kernel = np.zeros((kernel_size, kernel_size))
center = kernel_size // 2
for i in range(kernel_size):
    kernel[i, center] = 1
kernel = kernel / kernel.sum()  # Normalize kernel

# Set noise variance for Wiener filter
noise_var = 0.01

# Apply Wiener deconvolution
deblurred_img = wiener_deconvolution(img_blur, kernel, noise_var)

def remove_boundaries(image, padding_size):
    return image[padding_size:-padding_size, padding_size:-padding_size]

padding_size = kernel_size // 2
deblurred_img = remove_boundaries(deblurred_img, padding_size)

# Display the deblurred image
plt.figure()
plt.imshow(deblurred_img, cmap='gray')
plt.title('Deblurred Image')
plt.axis('off')

plt.show()
