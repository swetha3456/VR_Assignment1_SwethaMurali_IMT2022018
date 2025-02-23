import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess(image):
    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform K-Means using 4 clusters
    K = 4

    Z = grayscale.reshape((-1, 1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((grayscale.shape))

    # Separate the background from the remaining pixels
    segmented_image = label.reshape(grayscale.shape)

    max_white_pixels = 0
    max_mask = None

    # Iterate through each centroid to find the one with maximum pixels

    for i in range(K):
        # Create a binary mask for the current centroid:
        # white (255) where the condition is true, black (0) otherwise.
        mask = np.where(segmented_image == i, 255, 0).astype(np.uint8)

        # Count the number of white pixels (value 255)
        white_count = np.count_nonzero(mask == 255)

        # Check if this centroid has more white pixels than the previous maximum
        if white_count > max_white_pixels:
            max_white_pixels = white_count
            max_mask = mask

    # Remove black pixels at the bottom
    bottom_rows = int(max_mask.shape[0] * 0.07)
    max_mask[-bottom_rows:, :] = 255

    return max_mask

def coin_detection_canny(image):
    # Resize the image
    w = 0.15 * image.shape[1]
    h = 0.15 * image.shape[0]

    dim = (int(w), int(h))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Preprocess the image
    preprocessed_image = preprocess(image)

    # Detecting edges using the Canny edge detection method
    edges = cv2.Canny(preprocessed_image, 100, 200)

    # Thicken the edges
    kernel = np.ones((5, 5), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)

    count = 0

    # Find contours from the Canny edge image
    contours, _ = cv2.findContours(thick_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set threshold length for contours
    min_length = 60
    image_with_edges = image.copy()

    # Loop over contours and draw only those with an arc length above the threshold
    for cnt in contours:
        if cv2.arcLength(cnt, closed=True) > min_length:
            cv2.drawContours(image_with_edges, [cnt], -1, [0, 0, 255], thickness=2)
            count += 1

    return count, image_with_edges

def coin_segmentation(image):
    # Resize the image
    w = 0.15 * image.shape[1]
    h = 0.15 * image.shape[0]

    dim = (int(w), int(h))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Preprocess the image
    preprocessed_image = preprocess(image)

    # Blur the image and invert colors
    blur = cv2.medianBlur(preprocessed_image, 5)
    inverted = cv2.bitwise_not(blur)

    # Dilate the image to remove noise
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN,kernel, iterations = 2)

    # Finding sure background and foreground
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Detect and assign markers to each coin using connected components 
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1

    # Mark the unknown region with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers) # Applying Watershed algorithm

    # Segment the coins one by one
    segmented_coins = []

    for i in range(2, ret + 1):
        segmented = np.zeros_like(image)
        marker_mask = (markers == i)
        segmented[marker_mask] = image[marker_mask]
        segmented_coins.append(segmented)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    axes = axes.flatten()

    # Display the coins
    for i, img in enumerate(segmented_coins):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].axis("off")

    plt.tight_layout()

    image[markers == -1] = [255,0,0] # Outline the coins in the image

    return ret - 1, fig, image

def num_coins(image):
    """
    Function to return number of coins given an input image.
    """

    # Retrieve coin count by both detection and segmentation methods
    count_method1, _ = coin_detection_canny(image)
    count_method2, _, _ = coin_segmentation(image)

    assert(count_method1 == count_method2) # Ensure both methods return the same count

    return count_method1

if __name__ == "__main__":
    input_path = "input_images/coins.jpg"
    output_path = "output/"

    image = cv2.imread(input_path)
    print(f"Number of coins: {num_coins(image)}")

    _, outlined_image = coin_detection_canny(image)
    _, segmented_coins, watershed_outline = coin_segmentation(image)

    segmented_coins.savefig(output_path + "segmented_coins.png")
    cv2.imwrite(output_path + "canny_outline.png", outlined_image)
    cv2.imwrite(output_path + "watershed_outline.png", watershed_outline)

    print("Output images saved to directory", output_path)