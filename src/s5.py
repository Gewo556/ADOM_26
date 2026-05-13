import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def s5(image_path):
    try:
        image = plt.imread(image_path)
    except FileNotFoundError:
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(image, 'Plik nie znaleziony!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_cv = cv2.HOGDescriptor()
    hog_cv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    fd_16, hog_16 = hog(
        image, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=True, channel_axis=-1
    )
    hog_rescaled_16 = exposure.rescale_intensity(hog_16, in_range=(0, 10))

    rects_16, _ = hog_cv.detectMultiScale(
        gray, winStride=(8, 8), padding=(16, 16), scale=1.05
    )
    img_det_16 = image.copy()
    for (x, y, w, h) in rects_16:
        cv2.rectangle(img_det_16, (x, y), (x + w, y + h), (0, 255, 0), 3)

    fd_8, hog_8 = hog(
        image, orientations=8, pixels_per_cell=(8, 8),
        cells_per_block=(1, 1), visualize=True, channel_axis=-1
    )
    hog_rescaled_8 = exposure.rescale_intensity(hog_8, in_range=(0, 10))

    rects_8, _ = hog_cv.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.2,
    )
    img_det_8 = image.copy()
    for (x, y, w, h) in rects_8:
        cv2.rectangle(img_det_8, (x, y), (x + w, y + h), (0, 255, 0), 3)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Oryginał (v1)")
    axs[0, 1].imshow(hog_rescaled_16, cmap='gray')
    axs[0, 1].set_title("HOG 16x16 (v1)")
    axs[0, 2].imshow(img_det_16)
    axs[0, 2].set_title(f"Detekcja v1 (Ramki: {len(rects_16)})")

    axs[1, 0].imshow(image)
    axs[1, 0].set_title("Oryginał (v2)")
    axs[1, 1].imshow(hog_rescaled_8, cmap='gray')
    axs[1, 1].set_title("HOG 8x8 (v2)")
    axs[1, 2].imshow(img_det_8)
    axs[1, 2].set_title(f"Detekcja v2 (scale 1.2)\nRamki: {len(rects_8)}")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Długość wektora 16x16: {len(fd_16)}")
    print(f"Długość wektora 8x8:   {len(fd_8)}")