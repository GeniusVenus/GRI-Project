import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

color_map = {
    0: [0, 0, 0],  # Background (unlabeled - black)
    1: [255, 0, 0],  # Class 1 (building - #FF0000)
    2: [255, 105, 180],  # Class 2 (water - #FF69B4 )
    3: [65, 117, 5],  # Class 3 (tree - #417505)
    4: [126, 211, 33],  # Class 4 (grass - #7ED321)
    5: [169, 169, 169],  # Class 5 (road - #A9A9A9)
    6: [139, 87, 42]  # Class 6 (soil - #8B572A)
}


def convert_gray_to_rgb(img):
    img = cv2.merge([img, img, img])
    rgb_result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(0, 7):
        rgb_result[np.all(img == [i, i, i], axis=-1)] = color_map[i]
    return rgb_result


def convert_gray_to_rgb2(img):
    img = cv2.merge([img, img, img])
    rgb_result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(0, 6):
        rgb_result[np.all(img == [i, i, i], axis=-1)] = color_map[i + 1]
    return rgb_result


image_directory = "dataset/256_patches_splitted/test/images"
test_directory = "dataset/256_patches_splitted/test/masks"
for subdir in os.listdir(test_directory):
    result_dir = "image" + subdir[4:]
    image = cv2.imread(image_directory + "/" + result_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(test_directory + "/" + subdir)
    # result_rf = cv2.imread("results/RF/" + result_dir, 0)
    # result_xgb = cv2.imread("results/XGB/" + result_dir, 0)
    # result_svm = cv2.imread("results/SVM/" + result_dir, 0)
    result_unet = cv2.imread("results/UNET/" + result_dir, 0)
    result_vgg16_unet = cv2.imread("results/VGG16_UNET/" + result_dir, 0)
    # result_rf = convert_gray_to_rgb(result_rf)
    # result_xgb = convert_gray_to_rgb2(result_xgb)
    # result_svm = convert_gray_to_rgb(result_svm)
    result_unet = convert_gray_to_rgb2(result_unet)
    result_vgg16_unet = convert_gray_to_rgb2(result_vgg16_unet)
    for i in range(0, 7):
        mask[np.all(mask == [i, i, i], axis=-1)] = color_map[i]
    plt.subplot(1, 4, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title("Mask")
    plt.imshow(mask)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title("U-net")
    plt.imshow(result_unet)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title("VGG16-Unet")
    plt.imshow(result_vgg16_unet)
    plt.axis('off')
    # plt.subplot(1, 5, 1)
    # plt.title("Image")
    # plt.imshow(image)
    # plt.axis('off')
    # plt.subplot(1, 5, 2)
    # plt.title("Mask")
    # plt.imshow(mask)
    # plt.axis('off')
    # plt.subplot(1, 5, 3)
    # plt.title("RF")
    # plt.imshow(result_rf)
    # plt.axis('off')
    # plt.subplot(1, 5, 4)
    # plt.title("XGBoost")
    # plt.imshow(result_xgb)
    # plt.axis('off')
    # plt.subplot(1, 5, 5)
    # plt.title("SVM")
    # plt.imshow(result_svm)
    # plt.axis('off')
    plt.show()
# result_rf = cv2.imread("results/rf.tif", 0)
# result_xgb = cv2.imread("results/xgb.tif", 0)
# result_svm = cv2.imread("results/svm.tif", 0)
# plt.subplot(1, 3, 1)
# plt.title("Random Forest")
# plt.imshow(convert_gray_to_rgb(result_rf))
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.title("XGBoost")
# plt.imshow(convert_gray_to_rgb(result_xgb))
# plt.axis('off')
# plt.subplot(1, 3, 3)
# plt.title("SVM")
# plt.imshow(convert_gray_to_rgb(result_svm))
# plt.axis('off')
# plt.show()
# result_unet = cv2.imread("results/UNET.tif", 0)
# result_vgg16_unet = cv2.imread("results/VGG16_UNET.tif", 0)
# plt.subplot(1, 2, 1)
# plt.title("U-net")
# plt.imshow(convert_gray_to_rgb(result_unet))
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.title("VGG16-Unet")
# plt.imshow(convert_gray_to_rgb(result_vgg16_unet))
# plt.axis('off')
# plt.show()

# mask = cv2.imread("dataset/256_patches_splitted/train/masks/mask_patch_15.tif")
# result = cv2.imread("results/XGB_test_1.tif", 0)
# result = cv2.merge([result, result, result])
# rgb_result = np.zeros(result.shape, dtype=np.uint8)
# for i in range(0, 7):
#     rgb_result[np.all(result == [i, i, i], axis=-1)] = color_map[i]
#     mask[np.all(mask == [i, i, i], axis=-1)] = color_map[i]
# plt.subplot(1, 2, 1)
# plt.title("Mask")
# plt.imshow(mask)
# plt.subplot(1, 2, 2)
# plt.title("Segmented image")
# plt.imshow(rgb_result)
# plt.show()
