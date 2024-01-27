import cv2
import numpy as np
from skimage.filters import threshold_otsu
from keras.models import Model
from keras.applications.vgg16 import VGG16
import pandas as pd
import matplotlib.pyplot as plt

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in VGG_model.layers:
    layer.trainable = False
VGG_model.summary()
new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)


def convert_to_chromatic_R(image):
    image = image.astype(np.float32)
    green = image[:, :, 1]
    red = image[:, :, 0]
    blue = image[:, :, 2]
    all = red + green + blue
    return np.divide(red, all, out=np.zeros_like(red), where=all != 0)


def convert_to_chromatic_G(image):
    image = image.astype(np.float32)
    green = image[:, :, 1]
    red = image[:, :, 0]
    blue = image[:, :, 2]
    all = red + green + blue
    return np.divide(green, all, out=np.zeros_like(green), where=all != 0)


def convert_to_chromatic_B(image):
    image = image.astype(np.float32)
    green = image[:, :, 1]
    red = image[:, :, 0]
    blue = image[:, :, 2]
    all = red + green + blue
    return np.divide(blue, all, out=np.zeros_like(blue), where=all != 0)


def convert_to_gli(image):
    image = image.astype(np.float32)
    green = image[:, :, 1]
    red = image[:, :, 0]
    blue = image[:, :, 2]
    s1 = 2 * green - red - blue
    s2 = 2 * green + red + blue
    return np.divide(s1, s2, out=np.zeros_like(s1), where=s2 != 0)


def convert_to_ngrdi(image):
    image = image.astype(np.float32)
    green = image[:, :, 1]
    red = image[:, :, 0]
    s1 = green - red
    s2 = green + red
    return np.divide(s1, s2, out=np.zeros_like(s1), where=s2 != 0)


def apply_otsu_threshold(thresh):
    # Apply Otsu thresholding
    threshold_value = threshold_otsu(thresh)
    binary_image = thresh > threshold_value

    return binary_image


def feature_extraction(image):
    img = np.expand_dims(image, axis=0)
    features_VGG = new_model.predict(img)
    features = pd.DataFrame(features_VGG.reshape(-1, features_VGG.shape[3]))
    # Plot features from VGG
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            result = features_VGG[0, :, :, ix - 1]
            plt.imshow(result)
            ix += 1
    plt.show()
    # Plot another features
    features["Chromatic_R"] = apply_otsu_threshold(convert_to_chromatic_R(image)).reshape(-1)
    features["Chromatic_G"] = apply_otsu_threshold(convert_to_chromatic_G(image)).reshape(-1)
    features["Chromatic_B"] = apply_otsu_threshold(convert_to_chromatic_B(image)).reshape(-1)
    features["GLI"] = apply_otsu_threshold(convert_to_gli(image)).reshape(-1)
    features["NGRDI"] = apply_otsu_threshold(convert_to_ngrdi(image)).reshape(-1)
    plt.subplot(1, 5, 1)
    plt.title("Chr_R")
    plt.imshow(apply_otsu_threshold(convert_to_chromatic_R(image)))
    plt.axis('off')
    plt.subplot(1, 5, 2)
    plt.title("Chr_G")
    plt.imshow(apply_otsu_threshold(convert_to_chromatic_G(image)))
    plt.axis('off')
    plt.subplot(1, 5, 3)
    plt.title("Chr_B")
    plt.imshow(apply_otsu_threshold(convert_to_chromatic_B(image)))
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.title("GLI")
    plt.imshow(apply_otsu_threshold(convert_to_gli(image)))
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.title("NGRDI")
    plt.imshow(apply_otsu_threshold(convert_to_ngrdi(image)))
    plt.axis('off')
    plt.show()
    features.columns = features.columns.astype("str")
    return features


image = cv2.imread("dataset/256_patches/images/image_patch_17.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread("dataset/256_patches/masks/mask_patch_17.tif", 0)
plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask)
plt.show()
print(feature_extraction(image))
