import cv2
import numpy as np
import math
from PIL import Image
import io
import matplotlib.gridspec as gridspec


def read_image(file_path):
    """
    Reads an image using OpenCV.
    """
    return cv2.imread(file_path)

def Gray_image(image):
    """
    Converts an image's color to GRAYSCALE without using OpenCV's cvtColor function.
    """
    # Extract the R, G, B channels
    R = image[:, :, 2]
    G = image[:, :, 1]
    B = image[:, :, 0]
    
    # Calculate grayscale using the weighted method
    gray_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    return gray_image

def threshold(image):
    """
    Calculates the threshold value based on the mean intensity of the grayscale image.
    """
    gray_image = Gray_image(image)
    threshold_value = np.mean(gray_image)
    return threshold_value

def simple_halftoning(image):
    """
    Applies a simple halftoning technique without error diffusion.
    The function processes each pixel of the grayscale image and sets it to 0 (black) or 255 (white)
    based on a fixed threshold value of 128.
    """
    gray_image = Gray_image(image)
    row, column = gray_image.shape
    simple_halftoned_image = np.zeros_like(gray_image, dtype=np.uint8)

    for i in range(row):
        for j in range(column):
            # Threshold comparison
            simple_halftoned_image[i, j] = 255 if gray_image[i, j] > 128 else 0

    return simple_halftoned_image
def advanced_halftoning(image):
    """
    Applies an advanced halftoning technique using Floyd-Steinberg error diffusion.
    """
    gray_image = Gray_image(image)
    img_array = np.array(gray_image, dtype=np.float32)  # Float for error handling
    row, column = img_array.shape

    for i in range(row):
        for j in range(column):
            old_pixel = img_array[i, j]
            new_pixel = 255 if old_pixel > 128 else 0
            img_array[i, j] = new_pixel
            error = old_pixel - new_pixel

            # Distribute the error
            if j + 1 < column:
                img_array[i, j + 1] += error * 7 / 16
            if i + 1 < row:
                if j > 0:
                    img_array[i + 1, j - 1] += error * 3 / 16
                img_array[i + 1, j] += error * 5 / 16
                if j + 1 < column:
                    img_array[i + 1, j + 1] += error * 1 / 16

    # Clip the values to ensure they stay within [0, 255]
    img_array = np.clip(img_array, 0, 255)
    return img_array.astype(np.uint8)

import numpy as np
import matplotlib.pyplot as plt

def histogram(image):
    """
    Computes the histogram of an image and returns the plot as a PIL image.
    
    Parameters:
        image (numpy.ndarray): Input image (grayscale or RGB).
    
    Returns:
        hist (numpy.ndarray): Histogram data (256 bins for pixel intensities).
        hist_image (PIL.Image): Histogram visualization as a PIL image.
    """
    # Manually calculate the original histogram
    
    img_array=np.array(image)
    original_histogram = np.zeros(256, dtype=int)  # Array to store counts for 256 intensity levels
    for row in img_array:
        for pixel in row:
            original_histogram[pixel] += 1

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(original_histogram)
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255  # Normalize to range 0-255
    cdf_normalized = cdf_normalized.astype(np.uint8)  # Convert to integer values

    # Apply the CDF to equalize the image
    equalized_image = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            equalized_image[i, j] = cdf_normalized[img_array[i, j]]

    # Manually calculate the histogram of the equalized image
    equalized_histogram = np.zeros(256, dtype=int)
    for row in equalized_image:
        for pixel in row:
            equalized_histogram[pixel] += 1

       # Create a combined plot
    fig = plt.figure(figsize=(8, 14))  # Adjust figure size for better visualization
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 3])  # Allocate more height for the image

    # Original Histogram
    ax1 = plt.subplot(gs[0])
    ax1.bar(range(256), original_histogram, width=1, color="gray", edgecolor="black")
    ax1.set_title("Original Histogram")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(0, 255)

    # Equalized Histogram
    ax2 = plt.subplot(gs[1])
    ax2.bar(range(256), equalized_histogram, width=1, color="gray", edgecolor="black")
    ax2.set_title("Equalized Histogram")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 255)

    # Equalized Image
    ax3 = plt.subplot(gs[2])
    ax3.imshow(equalized_image, cmap="gray")
    ax3.set_title("Equalized Image")
    ax3.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot to a BytesIO stream
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()  # Close the plot to free resources

    # Move the buffer's cursor to the beginning
    buf.seek(0)

    # Convert the BytesIO stream to a PIL Image
    combined_image = Image.open(buf)
    


    # Convert the PIL Image to a NumPy array (if needed for your application)
    combined_image = np.array(combined_image)

    # Close the buffer after all operations are completed
    buf.close()

    return combined_image


def generateRowColumnSobelGradients():
    """Generates the x-component and y-component of Sobel operators."""
    rowGradient = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel kernel for row (horizontal)
    colGradient = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel kernel for column (vertical)
    return rowGradient, colGradient

def simple_edge_sobel(image):
    """
    Perform Sobel edge detection from scratch.

    Args:
        image (numpy array): Input grayscale image.

    Returns:
        result_rgb (numpy array): Sobel edge-detected image in RGB format.
    """
    # Convert to grayscale manually
    if len(image.shape) > 2:
        gray = Gray_image(image)
    else:
        gray = image

    rows, cols = gray.shape

    # Initialize result array
    result = np.zeros((rows, cols), dtype=float)

    # Sobel kernels
    row_gradient = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Horizontal gradient
    col_gradient = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical gradient

    # Pad the image
    padded_image = np.pad(gray, pad_width=1, mode='constant', constant_values=0)

    # Apply Sobel operator
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            subimage = padded_image[i-1:i+2, j-1:j+2]
            gx = np.sum(row_gradient * subimage)
            gy = np.sum(col_gradient * subimage)
            result[i-1, j-1] = math.sqrt(gx**2 + gy**2)

    # Normalize result to 0-255
    result = (result / result.max() * 255).astype(np.uint8)

    # Convert to RGB format
    result_rgb = np.stack((result, result, result), axis=-1)

    return result_rgb
def simple_edge_prewitt(image):
    """
    Perform Prewitt edge detection from scratch.

    Args:
        image (numpy array): Input image.

    Returns:
        magnitude (numpy array): Prewitt edge-detected grayscale image.
    """
    # Convert to grayscale manually
    if len(image.shape) > 2:
        gray = Gray_image(image)
    else:
        gray = image

    rows, cols = gray.shape

    # Initialize result array
    magnitude = np.zeros((rows, cols), dtype=float)

    # Prewitt kernels
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Horizontal gradient
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Vertical gradient

    # Pad the image
    padded_image = np.pad(gray, pad_width=1, mode='constant', constant_values=0)

    # Apply Prewitt operator
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            subimage = padded_image[i-1:i+2, j-1:j+2]
            px = np.sum(kernelx * subimage)
            py = np.sum(kernely * subimage)
            magnitude[i-1, j-1] = math.sqrt(px**2 + py**2)

    # Normalize to 0-255
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

    return magnitude

def simple_edge_kirsch(image):
    """
    Perform Kirsch edge detection from scratch.

    Args:
        image (numpy array): Input image.

    Returns:
        kirsch_magnitude (numpy array): Kirsch edge-detected grayscale image.
        kirsch_direction (numpy array): Direction of edges.
    """
    # Convert to grayscale manually
    if len(image.shape) > 2:
        gray = Gray_image(image)
    else:
        gray = image

    rows, cols = gray.shape

    # Kirsch kernels
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ]

    # Pad the image
    padded_image = np.pad(gray, pad_width=1, mode='constant', constant_values=0)
    kirsch_magnitude = np.zeros_like(gray, dtype=float)
    kirsch_direction = np.zeros_like(gray, dtype=int)

    # Apply Kirsch kernels
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            subimage = padded_image[i-1:i+2, j-1:j+2]
            responses = [np.sum(kernel * subimage) for kernel in kirsch_kernels]
            kirsch_magnitude[i-1, j-1] = max(responses)
            kirsch_direction[i-1, j-1] = np.argmax(responses)

    # Normalize magnitude to 0-255
    kirsch_magnitude = (kirsch_magnitude / kirsch_magnitude.max() * 255).astype(np.uint8)

    return kirsch_magnitude, kirsch_direction

def advanced_edge_homogeneity(image):
    """
    Applies advanced edge detection (homogeneity) manually.
    """
    threshold_value = 5
    gray_image = Gray_image(image)  # Use a manual grayscale conversion function
    height, width = gray_image.shape
    homogeneity_image = np.zeros((height, width), dtype=np.float32)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = gray_image[i, j]
            differences = [
                abs(float(center) - float(gray_image[i-1, j-1])),
                abs(float(center) - float(gray_image[i-1, j])),
                abs(float(center) - float(gray_image[i-1, j+1])),
                abs(float(center) - float(gray_image[i, j-1])),
                abs(float(center) - float(gray_image[i, j+1])),
                abs(float(center) - float(gray_image[i+1, j-1])),
                abs(float(center) - float(gray_image[i+1, j])),
                abs(float(center) - float(gray_image[i+1, j+1]))
            ]
            homogeneity_value = max(differences)
            homogeneity_image[i, j] = homogeneity_value if homogeneity_value > threshold_value else 0

    return homogeneity_image

def advanced_edge_difference(image):
    """
    Applies advanced edge detection (difference) manually.
    """
    threshold_value = 10
    gray_image = Gray_image(image)
    height, width = gray_image.shape
    difference_image = np.zeros((height, width), dtype=np.float32)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            difference1 = abs(float(gray_image[i-1, j-1]) - float(gray_image[i+1, j+1]))
            difference2 = abs(float(gray_image[i-1, j+1]) - float(gray_image[i+1, j-1]))
            difference3 = abs(float(gray_image[i, j-1]) - float(gray_image[i, j+1]))
            difference4 = abs(float(gray_image[i-1, j]) - float(gray_image[i+1, j]))
            max_difference = max(difference1, difference2, difference3, difference4)
            difference_image[i, j] = max_difference if max_difference > threshold_value else 0

    return difference_image

def advanced_edge_difference_of_Gaussians(image):
    """
    Applies advanced edge detection using manually defined Difference of Gaussians (7x7 and 9x9).
    
    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        
    Returns:
        numpy.ndarray: Output image after applying Difference of Gaussians, normalized to [0, 255].
    """
    # Ensure the image is grayscale
    
    
    if len(image.shape) == 3:  # Check if the image is colored (3 channels)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    height, width = gray_image.shape
    image_7x7 = np.zeros_like(gray_image, dtype=np.float32)
    image_9x9 = np.zeros_like(gray_image, dtype=np.float32)

    # Define Gaussian masks manually
    mask_7x7 = np.array([
        [0, 0, -1, -1, -1, 0, 0],
        [0, -2, -3, -3, -3, -2, 0],
        [-1, -3, 5, 5, 5, -3, -1],
        [-1, -3, 5, 16, 5, -3, -1],
        [-1, -3, 5, 5, 5, -3, -1],
        [0, -2, -3, -3, -3, -2, 0],
        [0, 0, -1, -1, -1, 0, 0]
    ], dtype=np.float32)

    mask_9x9 = np.array([
        [0, 0, 0, -1, -1, -1, 0, 0, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [-1, -3, -1, 9, 19, 9, -1, -3, -1],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, 0]
    ], dtype=np.float32)

    # Manually apply 7x7 and 9x9 masks
    for i in range(4, height - 4):  # Avoid boundaries for 9x9 mask
        for j in range(4, width - 4):
            subimage_7x7 = gray_image[i-3:i+4, j-3:j+4]  # Extract 7x7 region
            subimage_9x9 = gray_image[i-4:i+5, j-4:j+5]  # Extract 9x9 region

            # Ensure subimage matches the mask size
            if subimage_7x7.shape == mask_7x7.shape:
                image_7x7[i, j] = np.sum(mask_7x7 * subimage_7x7)
            if subimage_9x9.shape == mask_9x9.shape:
                image_9x9[i, j] = np.sum(mask_9x9 * subimage_9x9)

    # Calculate the Difference of Gaussian
    DifferenceOfGaussian = np.abs(image_7x7 - image_9x9)

    # Normalize the output to range [0, 255] and convert to uint8
    DifferenceOfGaussian = (DifferenceOfGaussian - DifferenceOfGaussian.min()) / \
                           (DifferenceOfGaussian.max() - DifferenceOfGaussian.min()) * 255
    
    return DifferenceOfGaussian.astype(np.uint8)

def advanced_edge_contrastBased(image):
    """
    Applies advanced edge detection (contrast-based) manually.
    """
    gray_image = Gray_image(image)

    # Gamma correction manually
    gamma = 0.5
    epsilon = 1e-10
    gamma_corrected = (gray_image / 255.0) ** gamma * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    # Laplacian edge-detection mask
    edge_mask = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=np.float32)

    # Smoothing mask
    smoothing_mask = np.ones((3, 3), dtype=np.float32) / 9

    # Apply edge detection manually
    height, width = gamma_corrected.shape
    image_edge = np.zeros_like(gamma_corrected, dtype=np.float32)
    image_smooth = np.zeros_like(gamma_corrected, dtype=np.float32)

    for i in range(1, height-1):
        for j in range(1, width-1):
            subimage = gamma_corrected[i-1:i+2, j-1:j+2]
            image_edge[i, j] = np.sum(edge_mask * subimage)
            image_smooth[i, j] = np.sum(smoothing_mask * subimage)

    # Avoid division by zero and compute contrast-enhanced edge image
    image_smooth += epsilon
    contrast_image_edge = image_edge / image_smooth

    # Normalize to 0-255
    contrast_image_edge = (contrast_image_edge / np.max(contrast_image_edge) * 255).astype(np.uint8)

    return contrast_image_edge
def advanced_edge_varianceBased(image):
    """
    Applies variance operator manually.
    """
    gray_image = Gray_image(image)  # Convert to grayscale manually
    height, width = gray_image.shape
    variance_edge_image = np.zeros_like(gray_image)

    for i in range(1, height-1):
        for j in range(1, width-1):
            neighborhood = gray_image[i-1:i+2, j-1:j+2]
            mean = np.sum(neighborhood) / 9
            variance = np.sum((neighborhood - mean)**2) / 9
            variance_edge_image[i, j] = variance

    return variance_edge_image

def advanced_edge_rangeBased(image):
    """
    Applies range operator manually.
    """
    gray_image = Gray_image(image)
    height, width = gray_image.shape
    range_edge_image = np.zeros_like(gray_image)

    for i in range(1, height-1):
        for j in range(1, width-1):
            neighborhood = gray_image[i-1:i+2, j-1:j+2]
            range_value = np.max(neighborhood) - np.min(neighborhood)
            range_edge_image[i, j] = range_value

    return range_edge_image

def high_bass_filtering(image):
    """
    Applies high-pass filter using manual convolution.
    """
    mask_high_pass = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)

    gray_image = Gray_image(image)
    height, width = gray_image.shape
    pad = 1  # Padding for 3x3 kernel
    padded_image = np.pad(gray_image, pad, mode='constant', constant_values=0)

    result_image = np.zeros_like(gray_image)

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+3, j:j+3]
            conv_value = np.sum(region * mask_high_pass)
            result_image[i, j] = max(0, min(255, conv_value))

    return result_image

def low_bass_filtering(image):
    """
    Applies low-pass filter using manual convolution.
    """
    mask_low_pass = np.array([[0, 1/6, 0],
                              [1/6, 2/6, 1/6],
                              [0, 1/6, 0]], dtype=np.float32)

    gray_image = Gray_image(image)
    height, width = gray_image.shape
    pad = 1  # Padding for 3x3 kernel
    padded_image = np.pad(gray_image, pad, mode='constant', constant_values=0)

    result_image = np.zeros_like(gray_image)

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+3, j:j+3]
            conv_value = np.sum(region * mask_low_pass)
            result_image[i, j] = max(0, min(255, conv_value))

    return result_image

def median_filtering(image):
    """
    Applies median filter using manual computation.
    """
    gray_image = Gray_image(image)
    kernel_size = 5
    height, width = gray_image.shape
    pad = kernel_size // 2
    padded_image = np.pad(gray_image, pad, mode='constant', constant_values=0)
    result_image = np.zeros_like(gray_image)

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            median_value = np.median(region)
            result_image[i, j] = median_value

    return result_image

def add_image(image):
    """
    Adds an image to itself manually.
    """
    gray_image = Gray_image(image)
    height, width = gray_image.shape
    result_image = np.zeros_like(gray_image)

    for i in range(height):
        for j in range(width):
            pixel_sum = gray_image[i, j] + gray_image[i, j]
            result_image[i, j] = max(0, min(255, pixel_sum))

    return result_image

def subtract_image(image):
    """
    Subtracts an image from itself manually.
    """
    gray_image = Gray_image(image)
    height, width = gray_image.shape
    result_image = np.zeros_like(gray_image)

    for i in range(height):
        for j in range(width):
            pixel_diff = gray_image[i, j] - gray_image[i, j]
            result_image[i, j] = max(0, min(255, pixel_diff))

    return result_image

def invert_image(image):
    """
    Inverts an image manually without built-in functions.
    """
    gray_image = Gray_image(image)  # Convert to grayscale manually
    height, width = gray_image.shape
    inverted_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            inverted_image[i, j] = 255 - gray_image[i, j]

    return inverted_image
def manual_segmentation(image, low_thresh, high_thresh, value=255):
    """
    Manual segmentation using thresholds.
    """
    gray_image = Gray_image(image)
    height, width = gray_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if low_thresh <= gray_image[i, j] <= high_thresh:
                mask[i, j] = value

    return mask

def histogram_peaks_segmentation(image, num_clusters=2, value=255):
    """
    Segmentation using histogram peaks manually.
    """
    gray_image = Gray_image(image)
    hist = np.zeros(256, dtype=int)

    # Calculate histogram manually
    for pixel in gray_image.flatten():
        hist[pixel] += 1

    # Find histogram peaks
    peaks = find_hist_peaks_manual(hist)

    if len(peaks) < num_clusters:
        raise ValueError(f"Not enough distinct peaks ({len(peaks)}) for {num_clusters} clusters")

    top_peaks = peaks[:num_clusters]
    thresholds = [0] + [(top_peaks[i] + top_peaks[i + 1]) // 2 for i in range(len(top_peaks) - 1)] + [255]

    # Create segmentation masks for each cluster
    cluster_masks = []
    height, width = gray_image.shape
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(num_clusters):
        cluster_mask = np.zeros((height, width), dtype=np.uint8)
        for x in range(height):
            for y in range(width):
                if thresholds[i] <= gray_image[x, y] < thresholds[i + 1]:
                    cluster_mask[x, y] = value
        cluster_masks.append(cluster_mask)

    return cluster_masks
def find_hist_peaks_manual(hist):
    """
    Manually find peaks in a histogram.
    """
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append(i)

    # Sort peaks by their intensity
    peaks.sort(key=lambda x: hist[x], reverse=True)
    return peaks

def histogram_valleys_segmentation(image, num_clusters=2, value=255):
    """
    Segments an image into clusters based on histogram valleys.

    Parameters:
        image (np.ndarray): Input grayscale image.
        num_clusters (int): Number of clusters (default: 2).
        value (int): Maximum pixel intensity for cluster masks (default: 255).

    Returns:
        np.ndarray: A single image with different intensity values for each cluster.
    """
    gray_image = Gray_image(image)
    hist = np.zeros(256, dtype=int)

    # Manually compute the histogram
    for pixel in gray_image.flatten():
        hist[pixel] += 1

    # Find peaks in the histogram
    peaks = find_hist_peaks_manual(hist)

    if len(peaks) < num_clusters + 1:
        raise ValueError(f"Not enough distinct peaks ({len(peaks)}) for {num_clusters + 1} valleys")

    # Sort peaks and find valleys
    peaks = sorted(peaks[:num_clusters + 1])  # Consider the first `num_clusters + 1` peaks
    valleys = []
    for i in range(len(peaks) - 1):
        valleys.append(find_hist_valley_manual(peaks[i:i + 2], hist))

    # Define thresholds based on valleys
    thresholds = [0] + valleys + [256]

    # Ensure `cluster_intensity` matches `num_clusters`
    cluster_intensity = [int((i + 1) * value / num_clusters) for i in range(num_clusters)]

    # Create a single segmented image with different intensities for clusters
    height, width = gray_image.shape
    segmented_image = np.zeros((height, width), dtype=np.uint8)

    for x in range(height):
        for y in range(width):
            pixel_value = gray_image[x, y]
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= pixel_value < thresholds[i + 1]:
                    if i < len(cluster_intensity):  # Avoid out-of-range index
                        segmented_image[x, y] = cluster_intensity[i]
                    break

    return segmented_image




def find_hist_valley_manual(peaks, hist):
    """
    Manually find the valley point between peaks in a histogram.

    Parameters:
        peaks (list): A pair of peak indices in the histogram.
        hist (list or np.ndarray): The histogram array.

    Returns:
        int: Index of the valley between the peaks.
    """
    start, end = peaks[0], peaks[1]
    min_val = float('inf')
    valley = start

    for i in range(start, end + 1):
        if hist[i] < min_val:
            min_val = hist[i]
            valley = i

    return valley


def histogram_adaptive_segmentation(image):
    """
    Adaptive Histogram Segmentation using the 2-pass technique.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # First pass: Calculate histogram and find peaks
    hist = np.zeros(256, dtype=int)
    for pixel in gray_image.flatten():
        hist[pixel] += 1

    peaks = find_hist_peaks_manual(hist)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks found for segmentation")

    # Set high and low threshold values
    low_threshold = peaks[0]
    high_threshold = peaks[1]

    print(f"First Pass - Low threshold: {low_threshold}, High threshold: {high_threshold}")

    # First segmentation using initial thresholds
    mask = np.zeros_like(gray_image)
    mask[(gray_image >= low_threshold) & (gray_image <= high_threshold)] = 255

    # Calculate the mean intensities for background and object
    background_mean = np.mean(gray_image[mask == 0]) if np.any(mask == 0) else 0
    object_mean = np.mean(gray_image[mask == 255]) if np.any(mask == 255) else 0
    print(f"First Pass - Background mean: {background_mean}, Object mean: {object_mean}")

    # Second pass: Use the means as new peaks to adjust thresholds
    new_low_threshold = int(min(background_mean, object_mean))
    new_high_threshold = int(max(background_mean, object_mean))

    print(f"Second Pass - New Low threshold: {new_low_threshold}, New High threshold: {new_high_threshold}")

    # Final segmentation using new thresholds
    final_mask = np.zeros_like(gray_image)
    final_mask[(gray_image >= new_low_threshold) & (gray_image <= new_high_threshold)] = 255

    return final_mask







