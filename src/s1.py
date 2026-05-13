import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import cv2
import numpy as np

def s1(image_path):
    image = plt.imread(image_path)

    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)


    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))


    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hog_cv = cv2.HOGDescriptor()
    hog_cv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    rects, weights = hog_cv.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )

    img_detected = image.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_detected, (x, y), (x + w, y + h), (0, 255, 0), 3)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    ax3.axis('off')
    ax3.imshow(img_detected)
    ax3.set_title('Detected person (HOG + SVM)')

    plt.tight_layout()
    plt.show()