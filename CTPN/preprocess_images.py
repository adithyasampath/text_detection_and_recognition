import numpy as np
import cv2
import imutils
from statistics import mode 
from PIL import Image

from skimage.filters import threshold_local

def get_dominant_colour(image):
    '''
    Args:
    image (np.ndarray): input binary image to fetch the most dominant color from
    
    Returns:
    colours: a list of the most common colours of the form [#0s, #255s]
    '''
    colors, count = np.unique(image.reshape(-1,image.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def remove_connected_components(image):
    '''
    Args:
    image (np.ndarray): binary image with connected components to be removed
    
    Returns:
    img_out (np.ndarray): binary image with connected components removed
    '''
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 150  
    img_out = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img_out[output == i + 1] = 255
    img_out = 255 - img_out
    return img_out

def preprocess_image(image, shape = (333, 75), kernel_size = (2,2)):
    '''
    Args:
    image (np.ndarray): 3 channel cv2 image of license plate crop to be preprocessed
    shape (Tuple): standardised size of the output image 
    kernel_size (Tuple): size of the dilation kernel (x,y)
    
    Returns:
    img_out (np.ndarray): pre-processed  3channel image in PIL format to be fed into tessaract
    '''
    resized_image = cv2.resize(image, shape)
    img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    color_counts = get_dominant_colour(img_binary)
    if mode(color_counts) == 255:
        img_binary = cv2.bitwise_not(img_binary)
    kernels =  np.ones(kernel_size, dtype=np.uint8)
    img_dilate = cv2.dilate(img_binary, kernels,iterations = 1)
    cleaned_image = remove_connected_components(img_dilate)
    img_out = Image.fromarray(np.uint8(cleaned_image))
    return img_out