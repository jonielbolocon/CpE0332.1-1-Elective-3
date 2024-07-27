%Bolocon, Joniel R.
% Read the original image
originalImage = imread('orange.png');

% Display the original image
figure;
imshow(originalImage);
title('Original Image');

% Create an image with only the red channel
redChannelImage = originalImage;
redChannelImage(:, :, 2) = 0; % Set green channel to 0
redChannelImage(:, :, 3) = 0; % Set blue channel to 0

figure;
imshow(redChannelImage);
title('Red Channel Image');

% Create an image with only the green channel
greenChannelImage = originalImage;
greenChannelImage(:, :, 1) = 0; % Set red channel to 0
greenChannelImage(:, :, 3) = 0; % Set blue channel to 0

figure;
imshow(greenChannelImage);
title('Green Channel Image');

% Create an image with only the blue channel
blueChannelImage = originalImage;
blueChannelImage(:, :, 1) = 0; % Set red channel to 0
blueChannelImage(:, :, 2) = 0; % Set green channel to 0

figure;
imshow(blueChannelImage);
title('Blue Channel Image');

% Convert the original image to grayscale
grayscaleImage = rgb2gray(originalImage);

figure;
imshow(grayscaleImage);
title('Grayscale Image');

% Display variable information
whos originalImage;
whos redChannelImage;
whos greenChannelImage;
whos blueChannelImage;
whos grayscaleImage;

% Save the images with high quality
imwrite(originalImage, 'original_orange_high_quality.jpg', 'jpg', 'Quality', 100);
imwrite(redChannelImage, 'red_channel_orange_high_quality.jpg', 'jpg', 'Quality', 100);
imwrite(greenChannelImage, 'green_channel_orange_high_quality.jpg', 'jpg', 'Quality', 100);
imwrite(blueChannelImage, 'blue_channel_orange_high_quality.jpg', 'jpg', 'Quality', 100);
imwrite(grayscaleImage, 'grayscale_orange_high_quality.jpg', 'jpg', 'Quality', 100);
