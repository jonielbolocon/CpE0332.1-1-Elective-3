% Read the image
img = imread('flower.jpg'); % Replace with the path to your image file


% Display the original image 
figure;
imshow(img); 
title('Original Image');


% Convert to grayscale if the image is RGB 
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end


% Display the grayscale image 
figure;
imshow(img_gray); 
title('Grayscale Image');


% Add blur to the image 
len = 21;
theta = 11;
psf = fspecial('motion', len, theta);
img_blur = imfilter(img_gray, psf, 'conv', 'circular');


% Display the blurred image 
figure; 
imshow(img_blur);
title('Motion Blurred Image');


% Filtering Techniques

% Gaussian filtering
h_gaussian = fspecial('gaussian', [5, 5], 1); 
img_gaussian_filtered = imfilter(img_blur, h_gaussian);


% Display the Gaussian filtered image 
figure; 
imshow(img_gaussian_filtered); 
title('Filtered Image (Gaussian)');


% Edge Detection
img_edges = edge(img_gray, 'Canny');


% Display the edge-detected image 
figure;
imshow(img_edges);
title('Edge Detected Image');


% Sharpening using unsharp masking 
img_sharpened = imsharpen(img_blur);


% Display the sharpened image 
figure; 
imshow(img_sharpened); 
title('Sharpened Image');


% Add Gaussian noise and remove it using median filter 
img_noisy = imnoise(img_gray, 'gaussian', 0.02); 
img_noisy_removed = medfilt2(img_noisy, [5, 5]);


% Display the noisy image 
figure;
imshow(img_noisy); 
title('Noisy Image');


% Display the noise-removed image 
figure; 
imshow(img_noisy_removed); 
title('Noise Removed Image');


% Deblurring 
estimated_nsr = 0.01;
img_deblurred = deconvwnr(img_blur, psf, estimated_nsr); 


% Display the deblurred image 
figure;
imshow(img_deblurred); 
title('Deblurred Image');