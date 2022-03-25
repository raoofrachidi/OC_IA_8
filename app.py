# Libraries
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from functions import infer

# Set parameters
class_name = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
img_height = 256
img_width = 512
n_classes = 8
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

# Flask app initialization
app = Flask(__name__)


@app.route('/')
def index():
    file_list = os.listdir('./static/images')
    return render_template('index.html', file_list=file_list)


@app.route('/predict', methods=['POST'])
def predict():
    # Return selected file
    file = request.form['file']
    image_path = str('./static/images/' + file)

    # Function to compute dice coefficient
    def dice_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
        return dice

    # Function to compute balanced cross entropy
    def balanced_cross_entropy(beta):
        def loss(y_true, y_pred):
            weight_a = beta * tensorflow.cast(y_true, tensorflow.float32)
            weight_b = (1 - beta) * tensorflow.cast(1 - y_true, tensorflow.float32)

            o = (
                        tensorflow.math.log1p(
                            tensorflow.exp(-tensorflow.abs(y_pred))
                        ) + tensorflow.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
            return tensorflow.reduce_mean(o)

        return loss

    def raw_iou(y_true, y_pred):
        results = []
        y_pred = y_pred > 0.3
        for i in range(0, y_true.shape[0]):
            intersect = np.sum(y_true[i, :, :] * y_pred[i, :, :])
            union = np.sum(y_true[i, :, :]) + np.sum(y_pred[i, :, :]) - intersect + 1e-7
            iou_metric = np.mean((intersect / union)).astype(np.float32)
            results.append(iou_metric)
        return np.mean(results)

    def iou(y_true, y_pred):
        iou_metric = tensorflow.numpy_function(raw_iou, [y_true, y_pred], tensorflow.float32)
        return iou_metric

    model = load_model(
        'final_model/final_model.h5',
        custom_objects={
            "loss": balanced_cross_entropy(0.3),
            "dice_coef": dice_coef,
            "iou": iou,
        }
    )

    # Predict from image
    seg_img = infer(
        model=model, inp=image_path, out_fname='./static/outputs/prediction.png',
        n_classes=n_classes, colors=class_colors,
        prediction_width=512, prediction_height=256,
        read_image_type=1)
    plt.imsave('./static/outputs/prediction.png', seg_img, cmap='nipy_spectral_r')

    return render_template('predict.html', image_data=file)


# Run app
if __name__ == '__main__':
    app.run(debug=True)
