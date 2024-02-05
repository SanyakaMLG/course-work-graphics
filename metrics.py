import numpy as np
import math
from skimage.io import imread


def psnr(original, compressed):
    max_pixel_value = 255.0

    mse = np.mean((original - compressed) ** 2)
    psnr_value = 20 * math.log10(max_pixel_value / math.sqrt(mse))

    return psnr_value


def ssim(original, compressed, dynamic_range=255):
    K1 = 0.01
    K2 = 0.03

    mu_original = np.mean(original)
    mu_compressed = np.mean(compressed)

    sigma_original = np.var(original)
    sigma_compressed = np.var(compressed)

    sigma_cov = np.cov(original.flatten(), compressed.flatten())[0, 1]

    C1 = (K1 * dynamic_range) ** 2
    C2 = (K2 * dynamic_range) ** 2
    numerator = (2 * mu_original * mu_compressed + C1) * (2 * sigma_cov + C2)
    denominator = (mu_original ** 2 + mu_compressed ** 2 + C1) * (sigma_original + sigma_compressed + C2)

    ssim_value = numerator / denominator

    return ssim_value


orig = imread('./test/test_camera_no_noise.jpeg')

compressed = imread('./test/test_camera_p_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('./test/test_camera_g_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('./test/test_blur_camera_no_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('./test/test_blur_camera_p_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('./test/test_blur_camera_g_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('./test/test_motion_camera_no_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('./test/test_motion_camera_p_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('./test/test_motion_camera_g_noise.jpeg')
print(ssim(orig, compressed))

compressed = imread('restored.jpg')
print(ssim(orig, compressed))

