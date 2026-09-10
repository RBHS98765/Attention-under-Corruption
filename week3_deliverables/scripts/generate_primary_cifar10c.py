"""Generate the 4 primary CIFAR-10-C corruptions for the Week 3 smoke test.

Ported verbatim from the official generation code:
  hendrycks/robustness, commit 8190fe3, ImageNet-C/create_c/make_cifar_c.py
(which is the code that produced the official Zenodo CIFAR-10-C archive).

Only the four primary corruptions are generated:
  brightness, contrast, defocus_blur, elastic_transform
at severities 1-5, laid out exactly like the official archive:
  [50000, 32, 32, 3] uint8, 5 consecutive blocks of 10000 (sev 1..5),
  plus labels.npy = [50000] with 5 identical blocks of the CIFAR-10 test labels.

Note: elastic_transform is stochastic (np.random displacement field); the
official archive itself is one draw. We seed np.random for reproducibility and
document this in the Week 3 report.
"""
import os
import time
import numpy as np
import cv2
from skimage.filters import gaussian
from skimage import color as sk_color
from scipy.ndimage import map_coordinates
import torchvision.datasets as dset

DATA_ROOT = os.environ.get("WEEK3_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")))
OUT = os.path.join(DATA_ROOT, "cifar10-c")
os.makedirs(OUT, exist_ok=True)
SEED = 0
np.random.seed(SEED)

# ---------- ported corruption helpers (verbatim from make_cifar_c.py) ----------
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def defocus_blur(x, severity=1):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255

def contrast(x, severity=1):
    c = [.75, .5, .4, .3, 0.15][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

def brightness(x, severity=1):
    c = [.05, .1, .15, .2, .3][severity - 1]
    x = np.array(x) / 255.
    x = sk_color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk_color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255

def elastic_transform(image, severity=1):
    IMSIZE = 32
    c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
         (IMSIZE*0.05, IMSIZE*0.2, IMSIZE*0.07),
         (IMSIZE*0.08, IMSIZE*0.06, IMSIZE*0.06),
         (IMSIZE*0.1, IMSIZE*0.04, IMSIZE*0.05),
         (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03)][severity - 1]
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

# ---------- main ----------
print("Loading CIFAR-10 test set...", flush=True)
ds = dset.CIFAR10(root=DATA_ROOT, train=False, download=False)
X = np.array([np.array(img) for img, _ in ds])
Y = np.array(ds.targets)
print("X:", X.shape, X.dtype, "Y:", Y.shape, flush=True)

CORRUPTIONS = {
    "brightness": brightness,
    "contrast": contrast,
    "defocus_blur": defocus_blur,
    "elastic_transform": elastic_transform,
}

t0 = time.time()
for name, fn in CORRUPTIONS.items():
    out = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
    for sev in range(1, 6):
        sl = (sev - 1) * 10000
        for i in range(10000):
            out[sl + i] = fn(X[i], severity=sev).astype(np.uint8)
        print(f"  {name} sev{sev} done ({time.time()-t0:.0f}s)", flush=True)
    assert out.shape == (50000, 32, 32, 3)
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()
    np.save(os.path.join(OUT, f"{name}.npy"), out)
    print(f"  saved {name}.npy", flush=True)

labels = np.tile(Y, 5)
np.save(os.path.join(OUT, "labels.npy"), labels)
print("saved labels.npy", labels.shape, flush=True)
print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
