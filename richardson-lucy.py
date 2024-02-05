import cv2
import numpy as np


def lucy_richardson_deconvolution(blurred_image, psf, iterations):
    estimated_sharp_image = blurred_image.astype(np.float64)

    conv_with_psf = cv2.filter2D(estimated_sharp_image, -1, psf)

    for _ in range(iterations):
        error = blurred_image / (conv_with_psf + 1e-10)

        estimated_sharp_image *= cv2.filter2D(error, -1, cv2.flip(psf, -1))

        conv_with_psf = cv2.filter2D(estimated_sharp_image, -1, psf)

    return estimated_sharp_image


blurred_image = cv2.imread('./test/test_blur_camera_p_noise.jpeg', 0)
psf = np.ones((9, 9)) / 81
estimated_sharp_image = lucy_richardson_deconvolution(blurred_image, psf, iterations=50)
cv2.imwrite('restored.jpg', estimated_sharp_image)
