import imageio as img
import numpy as np
import matplotlib.pyplot as plt

# Sobel X
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Sobel Y
sobelY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Load Image
im = img.imread('images.jpg')

# Convert to grayscale
if im.ndim == 3:
    im = im.mean(axis=2)

# Padding
imgPad = np.pad(im, pad_width=1, mode='constant', constant_values=0)

Gx = np.zeros_like(im)
Gy = np.zeros_like(im)

# Convolution
for y in range(1, imgPad.shape[0] - 1):
    for x in range(1, imgPad.shape[1] - 1):
        area = imgPad[y-1:y+2, x-1:x+2]
        Gx[y-1, x-1] = np.sum(area * sobelX)
        Gy[y-1, x-1] = np.sum(area * sobelY)

# Gradient magnitude
G = np.sqrt(Gx**2 + Gy**2)
G = (G / G.max()) * 255
G = G.astype(np.uint8)

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(im, cmap='gray')

plt.subplot(2,2,2)
plt.imshow(Gx, cmap='gray')

plt.subplot(2,2,3)
plt.imshow(Gy, cmap='gray')

plt.subplot(2,2,4)
plt.imshow(G, cmap='gray')

plt.show()
