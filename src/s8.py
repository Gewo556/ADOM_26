import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def s8(img_path):
    img_raw = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    hog_cv = cv2.HOGDescriptor()
    hog_cv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    gamma_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(len(gamma_values), 2, figsize=(16, 20))

    for idx, g in enumerate(gamma_values):
        img_gamma = adjust_gamma(img_rgb, g)
        gray_gamma = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2GRAY)

        rects, weights = hog_cv.detectMultiScale(
            gray_gamma, winStride=(8, 8), padding=(16, 16), scale=1.05
        )

        img_detected = img_gamma.copy()
        for x, y, w, h in rects:
            cv2.rectangle(img_detected, (x, y), (x + w, y + h), (0, 255, 0), 3)

        fd, hog_viz = hog(
            img_gamma,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            channel_axis=-1,
        )

        hog_rescaled = exposure.rescale_intensity(hog_viz, in_range=(0, 10))

        axes[idx, 0].imshow(img_detected)
        axes[idx, 0].set_title(
            f"Detekcja (Gamma={g}) | Wykryto osób: {len(rects)}"
        )
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(hog_rescaled, cmap="gray")
        axes[idx, 1].set_title(f"Mapa cech HOG (Gamma={g})")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.show()