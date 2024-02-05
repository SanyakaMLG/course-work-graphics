import numpy as np
from image_processing import image_to_numpy, plot_image, apply_filter, numpy_to_image


# convolution with gaussian filter
img_arr = image_to_numpy('panda.jpeg')
n, m = 13, 13
simple_kernel = np.ones((m, n)) / (n * m)
print(simple_kernel)

# resize image
img_arr = img_arr[::3, ::3, :]

# save image
img = numpy_to_image(img_arr)
img.save('panda_resized.jpeg')

# gaussian filter
sigma = 100
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, m)
xx, yy = np.meshgrid(x, y)
kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
kernel = kernel / np.sum(kernel)

print(kernel)


# apply simple filter
img_arr_1 = apply_filter(img_arr, simple_kernel)

# save image
img = numpy_to_image(img_arr_1)
img.save('panda_blurred.jpeg')

# apply gaussian filter
img_arr_2 = apply_filter(img_arr, kernel)

# save image
img = numpy_to_image(img_arr_2)
img.save('panda_blurred_gaussian.jpeg')
