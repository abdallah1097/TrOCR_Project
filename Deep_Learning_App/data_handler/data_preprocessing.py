import numpy as np
import tensorflow as tf
import cv2

class DataPreprocessor:
    """
    Class for preprocessing images for text extraction.
    """

    def __init__(self, image_size):
        """
        Initialize the DataPreprocessor.

        Args:
            image_size (tuple): Target image size (height, width).
        """
        self.image_size = image_size

    def preprocess(self, image):
        """
        Preprocess the input image for text extraction.

        Args:
            image (PIL.Image): Input image.

        Returns:
            tf.Tensor: Preprocessed image.
        """
        image = self.extract_text(image)

        # Resize the image
        image = tf.image.resize(image, self.image_size)
        image = image / 255.0

        return image

    def extract_text(self, img):
        """
        Extract text from the input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Cropped text region.
        """
        img = np.array(img)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform OTSU thresholding
        ret, thresh1 = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Creating a copy of the image
        im2 = img.copy()

        # Looping through the identified contours
        # Then rectangular part is cropped and passed on
        # to pytesseract for extracting text from it
        # Extracted text is then written into the text file
        if len(contours) == 0:
            # Couldn't detect any contours, pick the entire image
            return img
        elif len(contours) > 1:
            # Pick the biggest area contour; this is probably the correct one
            largest_area = 0
            largest_cnt = None

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_cnt = cnt
            contours = largest_cnt

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on the copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]

            return cropped
