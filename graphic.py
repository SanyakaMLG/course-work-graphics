import matplotlib.pyplot as plt
import numpy as np
import seaborn

from image_processing import plot_image
from skimage import data, restoration
from skimage.util import img_as_float
from skimage.io import imread, imshow, imsave
from metrics import psnr, ssim

original = img_as_float(imread('./test/test_camera_no_noise.jpeg'))
camera = img_as_float(imread('./test/test_blur_camera_g_noise.jpeg', as_gray=True))

psf = np.ones((9, 9)) / 81

ssims = [ssim(original[50:462, 50:462] * 255, camera[50:462, 50:462] * 255)]

for i in range(2, 51, 2):
    deconvolved = restoration.richardson_lucy(camera, psf, num_iter=i)

    # print('psnr=', psnr(original[50:462, 50:462] * 255, deconvolved[50:462, 50:462] * 255))
    # print('ssim=', ssim(original[50:462, 50:462] * 255, deconvolved[50:462, 50:462] * 255))
    ssims.append(ssim(original[50:462, 50:462] * 255, deconvolved[50:462, 50:462] * 255))

ax = seaborn.lineplot(y=ssims, x=np.arange(0, 51, 2))
ax.set(xlabel='Число итераций', ylabel='SSMS')
plt.grid(True)
plt.show()
