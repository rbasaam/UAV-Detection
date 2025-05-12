import os
import cv2

class DataLoader:
    def __init__(self, imageDir):
        self.imageDir = imageDir
        self.imagePaths = [os.path.join(imageDir, fname) for fname in os.listdir(imageDir) if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):

        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        return image

