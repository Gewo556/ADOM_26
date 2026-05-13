import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import cv2

def s3(image_path):
    image_raw = cv2.imread(image_path)

    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    roi = cv2.resize(image_raw[50:250, 100:200], (64, 128))

    def analyze_hog(img, orientations=9, ppc=8):
        fd, hog_image = hog(img,
                            orientations=orientations,
                            pixels_per_cell=(ppc, ppc),
                            cells_per_block=(2, 2),
                            visualize=True,
                            channel_axis=-1,
                            block_norm='L2-Hys',
                            transform_sqrt=True)

        hog_rescaled = exposure.rescale_intensity(hog_image,
                                                in_range=(0, np.percentile(hog_image, 98)))
        return fd, hog_rescaled


    # ==================== EKSPERYMENT 1: Liczba orientacji ====================
    orientations_list = [4, 9, 18]
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5.5))
    fig1.suptitle('Eksperyment 1: Liczba orientacji (Kierunkowość)', fontsize=16, y=0.96)

    for ax, orient in zip(axes1, orientations_list):
        fd, h_img = analyze_hog(roi, orientations=orient, ppc=8)
        ax.imshow(h_img, cmap='gray')
        ax.set_title(f"Orientations: {orient}\nRozmiar wektora: {len(fd)}", fontsize=11)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


    # ==================== EKSPERYMENT 2: Pixels Per Cell ====================
    ppc_list = [4, 8, 16]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.5))
    fig2.suptitle('Eksperyment 2: Pixels Per Cell (Gęstość siatki)', fontsize=16, y=0.96)

    for ax, ppc in zip(axes2, ppc_list):
        fd, h_img = analyze_hog(roi, orientations=9, ppc=ppc)
        ax.imshow(h_img, cmap='gray')
        ax.set_title(f"Cell size: {ppc}×{ppc}\nRozmiar wektora: {len(fd)}", fontsize=11)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()