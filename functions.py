import numpy as np
import random
import six
import cv2

class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]


# Function to predict
def infer(model=None, inp=None, out_fname=None, n_classes=None, prediction_width=None, prediction_height=None):
    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)), \
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, 1)

    assert (len(inp.shape) == 3 or len(inp.shape) == 1 or len(inp.shape) == 4), "Image should be h,w,3 "

    x = get_image_array(inp, prediction_width, prediction_height)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((128, 256, n_classes)).argmax(axis=2)

    seg_img = visualize_segmentation(
        pr,
        inp,
        n_classes=n_classes,
        prediction_width=prediction_width,
        prediction_height=prediction_height
    )

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def get_image_array(image_input, width, height):
    """ Load image array from input """
    img = None
    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        img = cv2.imread(image_input, 1)

    if img is not None:
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    return img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None, prediction_width=None, prediction_height=None):
    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes)

    if inp_img is not None:
        original_h = inp_img.shape[0]
        original_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if prediction_height is not None and prediction_width is not None:
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)

    return seg_img


def get_colored_segmentation_image(seg_arr, n_classes):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += (seg_arr_c * (class_colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += (seg_arr_c * (class_colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += (seg_arr_c * (class_colors[c][2])).astype('uint8')

    return seg_img
