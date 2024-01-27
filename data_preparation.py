import cv2
import os
import numpy as np
from patchify import patchify
from PIL import Image

img_dir = 'data/'
root_dir = 'dataset/'
patch_size = 256
if not os.path.exists(root_dir + "256_patches"):
    os.makedirs(root_dir + "256_patches")
else:
    if not os.path.exists(root_dir + "256_patches/images"):
        os.makedirs(root_dir + "256_patches/images")
    if not os.path.exists(root_dir + "256_patches/masks"):
        os.makedirs(root_dir + "256_patches/masks")

for path, subdirs, files in os.walk(img_dir):
    dirname = path.split(os.path.sep)[-1]
    if dirname.endswith('images'):
        # print(dirname)
        images = os.listdir(path)
        for x, image_name in enumerate(images):
            if image_name.endswith(".tif"):
                # print(image_name)
                image = cv2.imread(path + "/" + image_name, 1)
                SIZE_X = (image.shape[1] // patch_size) * patch_size
                SIZE_Y = (image.shape[0] // patch_size) * patch_size
                image = Image.fromarray(image)
                image = image.crop((0, 0, SIZE_X, SIZE_Y))
                image = np.array(image)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, :, :]
                        # single_patch_img = (single_patch_img.astype('float32')) / 255
                        # single_patch_img = scaler.fit_transform(
                        #     single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        single_patch_img = single_patch_img[0]
                        cv2.imwrite(root_dir + "256_patches/images/" + "image_patch_" + str(i) + str(j) + ".tif",
                                    single_patch_img)
    if dirname.endswith('masks_machine'):
        masks = os.listdir(path)
        for y, mask_name in enumerate(masks):
            # print(dirname)
            if mask_name.endswith(".png"):
                mask = cv2.imread(path + "/" + mask_name, 1)
                SIZE_X = (mask.shape[1] // patch_size) * patch_size
                SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                mask = np.array(mask)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]
                        single_patch_mask = single_patch_mask[0]
                        # single_patch_mask = (single_patch_mask.astype('float32')) / 255
                        cv2.imwrite(root_dir + "256_patches/masks/" + "mask_patch_" + str(i) + str(j) + ".tif",
                                    single_patch_mask)

import splitfolders

splitfolders.ratio('dataset/256_patches', 'dataset/256_patches_splitted', seed=1337, ratio=(0.6, 0.2, 0.2))

# from sklearn.preprocessing import MinMaxScaler
#
# scaler = MinMaxScaler()
# from keras.utils import to_categorical
#
#
# def preprocess_data(image, mask, num_class):
#     image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
#     mask = to_categorical(mask, num_class)
#     return (image, mask)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#
# def dataGenerator(image_path, mask_path, num_class):
#     # datagen_args = dict()
#     # classes = ["building", "water", "grass", "tree", "road", "soil"]
#     image_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
#     mask_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
#     image_generator = image_datagen.flow_from_directory(image_path, color_mode='rgb',
#                                                         batch_size=batch_size, class_mode=None,
#                                                         seed=seed)
#     mask_generator = mask_datagen.flow_from_directory(mask_path, color_mode='grayscale',
#                                                       batch_size=batch_size, class_mode=None, seed=seed)
#     data_generator = zip(image_generator, mask_generator)
#     for (image, mask) in data_generator:
#         image, mask = preprocess_data(image, mask, num_class)
#         yield (image, mask)

# train_gen = dataGenerator(train_image_path, train_mask_path, num_class=n_classes)
# val_gen = dataGenerator(val_image_path, val_mask_path, num_class=n_classes)
# test_gen = dataGenerator(test_image_path, test_mask_path, num_class=n_classes)
# X_train, Y_train = train_gen.__next__()
# X_val, Y_val = val_gen.__next__()
# X_test, Y_test = test_gen.__next__()
