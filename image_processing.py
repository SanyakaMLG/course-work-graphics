import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def image_to_numpy(img_path):
    # Read image
    img = Image.open(img_path)

    # Convert image to array
    img_arr = np.array(img)

    return img_arr


def numpy_to_image(img_arr):
    # Convert array to image
    img = Image.fromarray(img_arr.astype('uint8'))
    return img


def plot_image(img_arr):
    # Convert array to image
    img = numpy_to_image(img_arr)

    # Plot image
    plt.imshow(img)
    plt.show()


def apply_filter(img_arr, kernel):
    padding_h = kernel.shape[0] // 2
    padding_w = kernel.shape[1] // 2

    img_with_padding = np.zeros((img_arr.shape[0] + padding_h * 2, img_arr.shape[1] + padding_w * 2, 3))
    img_with_padding[padding_h:-padding_h, padding_w:-padding_w, :] = img_arr

    res_img = np.zeros(img_arr.shape)
    for i in range(res_img.shape[0]):
        for j in range(res_img.shape[1]):
            for k in range(3):
                res_img[i, j, k] = np.sum(img_with_padding[i:i + kernel.shape[0], j:j + kernel.shape[1], k] * kernel)

    return res_img
