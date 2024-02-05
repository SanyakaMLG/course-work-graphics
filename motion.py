import numpy as np
from image_processing import image_to_numpy, plot_image, apply_filter, numpy_to_image


# convolution with gaussian filter
img_arr = image_to_numpy('panda.jpeg')

# resize image
img_arr = img_arr[::3, ::3, :]

# save image
img = numpy_to_image(img_arr)
img.save('panda_resized.jpeg')

# motion blur filter
n = 9
simple_kernel = np.eye(n) / n


# apply simple filter
img_arr_1 = apply_filter(img_arr, simple_kernel)

# save image
img = numpy_to_image(img_arr_1)
img.save('panda_motion_blurred.jpeg')
