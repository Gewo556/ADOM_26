import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def s7(img_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    hog_cv = cv2.HOGDescriptor()
    hog_cv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    sigmas = [0, 1, 2, 4, 6, 8]
    fig, axes = plt.subplots(len(sigmas), 2, figsize=(16, 18))

    for idx, s in enumerate(sigmas):
        if s > 0:
            img_blurred = cv2.GaussianBlur(img_rgb, (0, 0), s)
        else:
            img_blurred = img_rgb.copy()

        gray_input = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2GRAY)

        rects, _ = hog_cv.detectMultiScale(
            gray_input, winStride=(8, 8), padding=(16, 16), scale=1.05
        )

        img_detected = img_blurred.copy()
        for x, y, w, h in rects:
            cv2.rectangle(img_detected, (x, y), (x + w, y + h), (0, 255, 0), 3)

        _, hog_viz = hog(
            img_blurred,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            channel_axis=-1,
        )

        hog_rescaled = exposure.rescale_intensity(hog_viz, in_range=(0, 10))

        axes[idx, 0].imshow(img_detected)
        axes[idx, 0].set_title(f"Detekcja (Sigma={s}) | Wykryto: {len(rects)}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(hog_rescaled, cmap="gray")
        axes[idx, 1].set_title(f"Mapa krawędzi HOG (Sigma={s})")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.show()