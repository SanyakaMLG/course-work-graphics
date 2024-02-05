import numpy as np
import skimage.util

from image_processing import numpy_to_image
from skimage.util import img_as_float
from skimage import data
from scipy.signal import convolve2d


def make_tests(img, filter, filename=None):
    filtered = convolve2d(img, filter, mode='same')
    numpy_to_image(filtered * 255).save('./test/test_' + filename + '_no_noise.jpeg')

    filtered_g_noisy = skimage.util.random_noise(filtered, var=0.01, mode='gaussian')
    numpy_to_image(filtered_g_noisy * 255).save('./test/test_' + filename + '_g_noise.jpeg')

    filtered_p_noisy = skimage.util.random_noise(filtered, mode='poisson')
    numpy_to_image(filtered_p_noisy * 255).save('./test/test_' + filename + '_p_noise.jpeg')


img = img_as_float(data.camera())
psf_blur = np.ones((9, 9)) / 81
psf_motion = np.eye(9, 9) / 9
make_tests(img, np.array([[1]]), 'camera')
make_tests(img, psf_blur, 'blur_camera')
make_tests(img, psf_motion, 'motion_camera')
