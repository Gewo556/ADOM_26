import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from IPython.display import display, clear_output
import time

def s2(image_path):
    image_raw = plt.imread(image_path)

    if len(image_raw.shape) == 2:
        image_raw = np.stack([image_raw] * 3, axis=-1)

    hog_cv = cv2.HOGDescriptor()
    hog_cv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    win_width, win_height = 64, 128
    scales = [1.0, 0.5]
    step_size = 80

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    try:
        for scale in scales:
            new_w = int(image_raw.shape[1] * scale)
            new_h = int(image_raw.shape[0] * scale)
            img_scaled = cv2.resize(image_raw, (new_w, new_h))

            for y in range(0, img_scaled.shape[0] - win_height, step_size):
                for x in range(0, img_scaled.shape[1] - win_width, step_size):
                    roi = img_scaled[y:y + win_height, x:x + win_width]

                    fd, hog_roi_viz = hog(
                        roi,
                        orientations=8,
                        pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1),
                        visualize=True,
                        channel_axis=-1
                    )

                    ax1.clear()
                    clone = img_scaled.copy()
                    cv2.rectangle(clone, (x, y), (x + win_width, y + win_height), (0, 255, 0), 5)
                    ax1.imshow(clone)
                    ax1.set_title(f"Skala: {int(scale*100)}% | Skanowanie okno 64x128")
                    ax1.axis('off')

                    ax2.clear()
                    hog_rescaled = exposure.rescale_intensity(hog_roi_viz, in_range=(0, 10))
                    ax2.imshow(hog_rescaled, cmap='gray')
                    ax2.set_title("To 'widzi' SVM (HOG)")
                    ax2.axis('off')

                    display(fig)
                    clear_output(wait=True)
                    time.sleep(0.001)
    except KeyboardInterrupt:
        print("Zatrzymano animację.")

    plt.close(fig)

    gray = cv2.cvtColor(image_raw, cv2.COLOR_RGB2GRAY)

    rects, weights = hog_cv.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )

    img_detected = image_raw.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_detected, (x, y), (x + w, y + h), (0, 255, 0), 3)

    _, hog_full_viz = hog(image_raw, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    hog_full_rescaled = exposure.rescale_intensity(hog_full_viz, in_range=(0, 10))

    fig_final, (f1, f2, f3) = plt.subplots(1, 3, figsize=(18, 6))
    f1.imshow(image_raw); f1.set_title('1. Obraz wejściowy'); f1.axis('off')
    f2.imshow(hog_full_rescaled, cmap='gray'); f2.set_title('2. Mapa HOG (100% skali)'); f2.axis('off')
    f3.imshow(img_detected); f3.set_title('3. Wynik detekcji (Wszystkie skale)'); f3.axis('off')

    plt.tight_layout()
    plt.show()