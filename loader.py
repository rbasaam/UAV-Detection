"""
Data loading module for UAV Detection.
This module provides functionality for loading and previewing images.
"""

import os
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DataLoader")


class DataLoader:
    """
    Handles image loading and preprocessing for object detection.
    
    Provides indexing, iteration, and preview functionality for
    images stored in a directory.
    """
    
    def __init__(self, imageDir):
        """
        Initialize the DataLoader with a directory of images.
        
        Args:
            imageDir (str): Path to directory containing images
        """
        self.imageDir = imageDir
        self.imagePaths = [
            os.path.join(imageDir, fname) 
            for fname in os.listdir(imageDir) 
            if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))
        ]
        log.info(f"Loaded {len(self.imagePaths)} images from {imageDir}")
    
    def __len__(self):
        """
        Return the number of images in the dataset.
        
        Returns:
            int: Number of images
        """
        return len(self.imagePaths)

    def __getitem__(self, idx):
        """
        Get an image by index.
        
        Args:
            idx (int): Index of the image to retrieve
            
        Returns:
            numpy.ndarray: Image as RGB numpy array
        """
        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath)
        if image is None:
            log.error(f"Could not load image at {imagePath}")
            return None
            
        # Convert BGR to RGB for compatibility with most ML models
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def preview(self, idx):
        """
        Preview an image from the dataset.
        
        Displays the image in a window if display is available,
        otherwise provides information about the image.
        
        Args:
            idx (int): Index of the image to preview
            
        Returns:
            bool: True if preview was successful, False otherwise
        """
        if idx < 0 or idx >= len(self.imagePaths):
            log.error(f"Index {idx} is out of range. Valid range: 0-{len(self.imagePaths)-1}")
            return False
        
        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath)
        
        if image is None:
            log.error(f"Could not load image at {imagePath}")
            return False
        
        # Get basic image information
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        log.info(f"Image: {os.path.basename(imagePath)}")
        log.info(f"Dimensions: {width}x{height}, {channels} channels")
        
        try:
            # Try to display the image (might fail in headless environments)
            window_name = f"Preview: {os.path.basename(imagePath)}"
            cv2.imshow(window_name, image)
            log.info("Press any key to close the preview window...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
            return True
        except Exception as e:
            log.warning(f"Could not display image: {str(e)}")
            # Fallback: Save a small version of the image to a temp file
            try:
                temp_path = os.path.join('/tmp', f"preview_{os.path.basename(imagePath)}")
                cv2.imwrite(temp_path, image)
                log.info(f"Image saved to {temp_path} for viewing")
                return True
            except:
                log.error("Could not save preview image")
                return False
    
    def getImagePath(self, idx):
        """
        Get the file path for an image.
        
        Args:
            idx (int): Index of the image
            
        Returns:
            str: Path to the image file or None if index is invalid
        """
        if 0 <= idx < len(self.imagePaths):
            return self.imagePaths[idx]
        return None
    
    def getImageInfo(self, idx):
        """
        Get information about an image without loading pixel data.
        
        Args:
            idx (int): Index of the image
            
        Returns:
            dict: Dictionary with image information or None if index is invalid
        """
        if idx < 0 or idx >= len(self.imagePaths):
            return None
            
        imagePath = self.imagePaths[idx]
        try:
            filename = os.path.basename(imagePath)
            filesize = os.path.getsize(imagePath) / 1024  # KB
            infoDict = {
                'filename': filename,
                'path': imagePath, 
                'size_kb': filesize
            }
            print(f"Image Info:\n{infoDict}")
            return infoDict
        except:
            return None

