# Import standard libraries
import os
import cv2
import numpy as np
from io import BytesIO

# Brightness detector functions embedded directly in the Nuclio function
def calculate_brightness(image_path=None, roi=None, image_data=None):
    """
    Calculate the average brightness in an image or ROI.
    
    Args:
        image_path (str, optional): Path to the image file
        roi (tuple, optional): Region of interest as (x, y, width, height)
        image_data (bytes, optional): Raw image data as bytes
        
    Returns:
        float: Average brightness value (0-255)
    """
    # Load the image either from path or from data
    if image_data is not None:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_path is not None:
        # Read the image from file
        img = cv2.imread(image_path)
    else:
        raise ValueError("Either image_path or image_data must be provided")
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # If ROI is specified, crop the image
    if roi:
        x, y, w, h = roi
        try:
            roi_img = gray[y:y+h, x:x+w]
            avg_brightness = float(np.mean(roi_img))
        except IndexError:
            raise IndexError(f"ROI {roi} is outside image boundaries {gray.shape}")
    else:
        # Use the whole image if no ROI specified
        avg_brightness = float(np.mean(gray))
    
    return avg_brightness