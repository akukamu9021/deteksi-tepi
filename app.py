import imageio as img
import numpy as np
import matplotlib.pyplot as plt

# Kernel Sobel X
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Kernel Sobel Y
sobelY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Baca citra grayscale
image = img.imread('Salt-Noise.jpg', mode='F')

# Padding gambar (konstanta = 0)
imgPad = np.pad(image, mode='constant', pad_width=1, constant_values=0)

# Matriks penampung hasil gradien
Gx = np.zeros_like(imgPad)
Gy = np.zeros_like(imgPad)

# Konvolusi manual Sobel
for y in range(1, imgPad.shape[0] - 1):
    for x in range(1, imgPad.shape[1] - 1):
        area = imgPad[y-1:y+2, x-1:x+2]
        Gx[y-1, x-1] = np.sum(area * sobelX)
        Gy[y-1, x-1] = np.sum(area * sobelY)

# Magnitudo gradien
G = np.sqrt(Gx**2 + Gy**2)

# Normalisasi ke rentang 0–255
G = (G / G.max()) * 255
G = np.clip(G, 0, 255)
G = G.astype(np.uint8)

# -----------------------------
# BASIC THRESHOLDING (SEGMENTASI)
# -----------------------------
T = 80   # nilai ambang (bisa diuji coba)
binary = np.where(G > T, 255, 0).astype(np.uint8)

# Tampilkan hasil
plt.figure(figsize=(10,10))

plt.subplot(2,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(Gx, cmap='gray')
plt.title("Gradien X (SobelX)")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(Gy, cmap='gray')
plt.title("Gradien Y (SobelY)")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(G, cmap='gray')
plt.title("Magnitude (Edge Result)")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(binary, cmap='gray')
plt.title("Segmentasi (Thresholding)")
plt.axis("off")

plt.tight_layout()
plt.show()
