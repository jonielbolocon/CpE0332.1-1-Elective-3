%Group 2 - Exercise 2

%Read the Image 'flower.jpg'
img = imread('flower.jpg'); % No need to specify path since it is in MATLAB Drive

%Display the Image
figure(1);
imshow(img);
title('Original Image')

%Get Image Dimensions (rows, columns, color channels)
[rows, cols, channels] = size(img);
disp(['Image Size: ', num2str(rows), 'x', num2str(cols), 'x', num2str(channels)]);

%Check Color Model (Grayscale or RGB)
if channels == 1
    disp('Color Model: Grayscale');
else
    disp('Color Model: RGB');
end

%Access Individual Pixels (example: center pixel)
center_row = floor(rows/2) + 1;
center_col = floor(cols/2) + 1;
center_pixel = img(center_row, center_col, :);
disp(['Center Pixel Value: ', num2str(center_pixel)]);

%Basic Arithmetic Operations (Add Constant Value To All Pixels)
brightened_img = img + 50;
figure(2);
imshow(brightened_img);
title('Brightened Image');

%Basic Geometric Operation (Flipping Image Horizontally)
flipped_img = fliplr(img);
figure(3);
imshow(flipped_img);
title('Horizontally Flipped Image');