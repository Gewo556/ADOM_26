import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import time

def s6(img_path):
    image_raw = cv2.imread(img_path)
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image_res = cv2.resize(image_raw, (800, 600))
    gray = cv2.cvtColor(image_res, cv2.COLOR_RGB2GRAY)

    hog_cv = cv2.HOGDescriptor()
    hog_cv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    scenarios = [
        {
            "name": "Precyzyjny (Gęsty)",
            "winStride": (4, 4),
            "scale": 1.03,
            "color": (0, 255, 0)
        },
        {
            "name": "Zrównoważony (Szybszy)",
            "winStride": (4, 4),
            "scale": 1.1,
            "color": (255, 0, 0)
        },
        {
            "name": "Okno przesuwa się zbyt rzadko",
            "winStride": (16, 16),
            "scale": 1.03,
            "color": (255, 165, 0)
        },
        {
            "name": "Duża liczba poziomów piramidy (Zbyt gęsty skok)",
            "winStride": (4, 4),
            "scale": 1.01,
            "color": (255, 255, 0)
        }
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))

    for i, spec in enumerate(scenarios):
        levels = int(np.log(64 / 800) / np.log(1 / spec["scale"]))

        start_time = time.time()
        rects, weights = hog_cv.detectMultiScale(
            gray,
            winStride=spec["winStride"],
            padding=(8, 8),
            scale=spec["scale"]
        )
        duration = time.time() - start_time

        res_img = image_res.copy()
        for (x, y, w, h) in rects:
            cv2.rectangle(res_img, (x, y), (x + w, y + h), spec["color"], 3)

        axes[i, 0].imshow(res_img)
        axes[i, 0].set_title(f"Scenariusz: {spec['name']}\nCzas: {duration:.3f}s | Wykryto: {len(rects)} | Poziomy: ~{levels}")
        axes[i, 0].axis('off')

        if len(rects) > 0:
            widths = [w for (x, y, w, h) in rects]
            axes[i, 1].hist(widths, bins=range(50, 400, 20), color='teal', alpha=0.7, edgecolor='black')
            axes[i, 1].set_xlabel("Szerokość okna (skala w px)")
            axes[i, 1].set_ylabel("Liczba detekcji")
            axes[i, 1].set_title(f"Rozkład szerokości okien")
        else:
            axes[i, 1].text(0.5, 0.5, "Brak detekcji", ha='center', va='center', fontsize=14)
            axes[i, 1].set_title("Rozkład szerokości okien (Brak danych)")

    plt.tight_layout()
    plt.show()