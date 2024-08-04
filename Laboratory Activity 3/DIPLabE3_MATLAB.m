% Read an image	 
img = imread('flower.jpg'); 

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

% Contrast enhancement using imadjust 
img_contrast_enhanced = imadjust(img_gray); 

% Display the contrast-enhanced image 
figure; 
imshow(img_contrast_enhanced); 
title('Contrast Enhanced Image (imadjust)'); 

% Histogram equalization 
img_histeq = histeq(img_gray); 

% Display the histogram equalized image 
figure; 
imshow(img_histeq); 
title('Equalized Image'); 

% Filtering using average filter 
h_avg = fspecial('average', [5, 5]); 
img_avg_filtered = imfilter(img_gray, h_avg); 

% Display the average filtered image 
figure; 
imshow(img_avg_filtered); 
title('Filtered Image (Average)'); 

% Filtering using median filter 
img_median_filtered = medfilt2(img_gray, [5, 5]); 

% Display the median filtered image 
figure; 
imshow(img_median_filtered); 
title('Filtered Image (Median)'); 

% Display histograms for comparison 

% Grayscale histogram 
figure; 
imhist(img_gray); 
title('Histogram of Grayscale'); 


% Enhanced histogram (imadjust) 
figure; 
imhist(img_contrast_enhanced); 
title('Histogram of Enhanced Image'); 

% Equalized histogram 
figure; 
imhist(img_histeq); 
title('Histogram of Equalized Image'); 

% Histogram (Average Filtered) 
figure; 
imhist(img_avg_filtered); 
title('Histogram of Average Filtered)'); 

% Histogram (Median Filtered) 
figure; 
imhist(img_median_filtered); 
title('Histogram of Median Filtered)'); 