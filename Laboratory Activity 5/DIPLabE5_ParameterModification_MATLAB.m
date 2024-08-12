% Global Image Thresholding using Otsu's Method
% Load image and convert to grayscale
img = imread('flower.jpg');
figure(1);
imshow(img);
title('Original Image');

gray_img = rgb2gray(img);  % Convert to grayscale for thresholding

% Calculate threshold using graythresh
level = graythresh(gray_img);

% Convert into binary image using the computed threshold
bw = imbinarize(gray_img, level);

% Display the original image and the binary image
figure(2);
imshowpair(img, cat(3, bw, bw, bw), 'montage'); % Convert binary to RGB for display
title('Original Image (left) and Binary Image (right)');

% Multi-level Thresholding using Otsu's Method
% Calculate multiple thresholds using multithresh
levels = multithresh(gray_img, 2);  % Adjust number of levels as needed

% Segment the image into regions using the imquantize function
seg_img = imquantize(gray_img, levels);

% Display the original image and the segmented image with RGB color
seg_img_rgb = label2rgb(seg_img); % Convert segmented image to RGB for display
figure(3);
imshowpair(img, seg_img_rgb, 'montage');
title('Original Image (left) and Segmented Image (right)');

% Global Histogram Thresholding using Otsu's Method
% Calculate a 16-bin histogram for the grayscale image
[counts, x] = imhist(gray_img, 16);

% Plot the histogram
figure(4); % Create a new figure for the histogram
bar(x, counts);
title('16-bin Histogram');
xlabel('Intensity Value');
ylabel('Pixel Count');

% Compute a global threshold using the histogram counts
T = otsuthresh(counts);

% Create a binary image using the computed threshold
bw = imbinarize(gray_img, T);

% Display the original image and the binary image
figure(5);
imshowpair(img, cat(3, bw, bw, bw), 'montage'); % Convert binary to RGB for display
title('Original Image (left) and Binary Image');

% Region-Based Segmentation
% Convert the image to grayscale for k-means clustering
bw_img2 = rgb2gray(img);

% Segment the image into three regions using k-means clustering
[L, centers] = imsegkmeans(bw_img2, 3);
B = labeloverlay(img, L); % Use original RGB image for label overlay
figure(6);
imshow(B);
title('Labeled Image');

% Connected-Component Labeling
% Compute the binary image using a threshold
threshold = 0.5; % Adjust as needed
bin_img2 = imbinarize(rgb2gray(img), threshold);

% Label the connected components
[labeledImage, numberOfComponents] = bwlabel(bin_img2);

% Display the number of connected components
disp(['Number of connected components: ', num2str(numberOfComponents)]);

% Assign a different color to each connected component
coloredLabels = label2rgb(labeledImage, 'hsv', 'k', 'shuffle');

% Display the labeled image
figure(7);
imshow(coloredLabels);
title('Labeled Image');


% Parameter Modifications
% Add noise to the RGB image and segment using Otsu's method
img_noise = imnoise(img, 'salt & pepper', 0.09);

% Calculate thresholds using multithresh
levels = multithresh(img_noise);

% Segment the noisy grayscale image into two regions using imquantize
seg_img_noise = imquantize(img_noise, level);

% Display the original noisy image and the segmented image with RGB color
figure(8);
imshowpair(img_noise, seg_img_rgb_noise, 'montage');
title({'Original Image with Noise (left) and';' Segmented Image with Noise (right)'});

% Segment the original RGB image into regions using k-means clustering
L = imsegkmeans(img, 2);
B = labeloverlay(img, L);
figure(9);
imshow(B);
title('Labeled Image');

% Create and apply Gabor filters
wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength, orientation);

% Convert the image to grayscale
bw_RGB = im2gray(im2single(img));

% Filter the grayscale image using the Gabor filters
gabormag = imgaborfilt(bw_RGB, g);
figure(10);
montage(gabormag, "Size", [4 6]);

% Smooth each filtered image to remove local variations
for i = 1:length(g)
    sigma = 0.5 * g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i), 3 * sigma);
end
figure(11);
montage(gabormag, "Size", [4 6]);

% Get the x and y coordinates of all pixels in the input image
[nrows, ncols] = size(bw_RGB);
[X, Y] = meshgrid(1:ncols, 1:nrows);
featureSet = cat(3, bw_RGB, gabormag, X, Y);

% Segment the image into regions using k-means clustering with the supplemented feature set
L2 = imsegkmeans(featureSet, 2, "NormalizeInput", true);
C = labeloverlay(img, L2); % Use original RGB image for label overlay
figure(12);
imshow(C);
title("Labeled Image with Additional Pixel Information");