import pytesseract
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def tesseract(  image,
                config  = '--psm 7 -c tessedit_char_whitelist=0123456789ZXCVBNMLKJHGFDSAPOIUYTREWQ.%',
                lang    = 'eng'):
    '''Returns a string corresponding to the text in the image
    Args:
        image : PIL image 3 channel
    Returns
        A string corresponding to the text in the image
    '''
    image = image.convert('L')
    return pytesseract.image_to_string(image, config=config, lang=lang).upper()
