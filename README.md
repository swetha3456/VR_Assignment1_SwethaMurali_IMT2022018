# VR_Assignment1_SwethaMurali_IMT2022018

## How to Run

To execute the script, use the following commands:

```sh
python3 coin_analysis.py
python3 panorama_image_stitching.py
```

The output files are generated and saved in the `output` directory.

## Steps

### 1. Coin Detection and Segmentation

**Preprocessing**

- Resize the image, convert to grayscale, and perform K-Means using 4 clusters.

![image](https://github.com/user-attachments/assets/d350df48-0a00-4bf0-be4d-d8417c166632)

- Choose the cluster with the maximum number of pixels - this is the background. Separate this from the remaining clusters.

![image](https://github.com/user-attachments/assets/28d7e8ae-6b51-4983-91a6-7c451485cf35)



**Canny Edge Detection and Outlining Coins**

- Perform Canny edge detection.

![image](https://github.com/user-attachments/assets/f60ca057-fdb6-4fc3-8f2d-a8725d6f74da)

- Dilate the edges and filter contours with length above a threshold. Use these contours to outline the coins in the image. The number of contours gives us the number of coins.

![image](https://github.com/user-attachments/assets/dc1ca2fa-1b79-430e-af7e-25133a878301)



**Coin Segmentation**
- Repeat the preprocessing steps.
- Blur the image and invert the colors.
 
![image](https://github.com/user-attachments/assets/a87a9d0c-1d91-46f4-996a-f928a92bb213)

- Dilate the edges to remove noise.

![image](https://github.com/user-attachments/assets/5cb9c7dd-112e-47ae-84d8-8a1ca91ea1b6)

- Separate the foreground and background using segmentation. Use markers to demarcate approximate boundaries for each coin.

![image](https://github.com/user-attachments/assets/b3f960ec-eefb-47b5-9568-9d070873265a)

- Apply the Watershed algorithm to segment each coin.

![image](https://github.com/user-attachments/assets/f7560cdb-763c-4680-871a-3785b6b3b4a9)

- Draw outlines around each coin using the Watershed boundaries.

![image](https://github.com/user-attachments/assets/36128a27-2d22-469b-aeb6-b33655af9612)

### 2. Panorama Image Stitching

- Generate SIFT descriptors and find top matches.

![image](https://github.com/user-attachments/assets/28b307bf-d0b0-4f83-a7f7-80dc3ebfd137)

- Warp and blend the images to get the resultant panorama.

![image](https://github.com/user-attachments/assets/c93a4d71-cc55-4745-9aea-4ea1f10975d6)

