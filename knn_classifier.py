## different libraries ##
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from time import time

## arguments to take from console
parser = argparse.ArgumentParser(
    description="Running K-nn classifier on hand written Hebrew letters ",
    epilog="And that's it :) ... ")

parser.add_argument('--path', type=str, metavar='<data file name>',
                    help="Here you enter the datas' file path")
parser.add_argument('--metric', type=int, metavar='<which metric to use>',
                    help="1 is for Chi Squared and 0 for Euclidean distance , "
                         "if you choose this then you have to use --k")
parser.add_argument('--k', type=int, metavar='<number of k neighbors>',
                    help="choose a k from 1 to 15,"
                         "if you choose this then you have to use --metric")
args = parser.parse_args()

## Reading the data
if not args.path:
    args.path = "C:/Users/yoooo/Desktop/hhd_dataset"
training_path = args.path + "/TRAIN/"
testing_path = args.path + "/TEST/"
labels = [str(x) for x in range(27)]
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def calculate_time(Starttime):
    """
    calculates the time from start to current time and returns it in secedes
    :param Starttime: this is the time from which we want to calculate how much time has passed since this time
    :returns:the current time
    """
    return round(time() - Starttime, 2)


def display(img, cmap=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)


def image_processing(img):
    """
    making the padding for each image and then resizing it
    """
    desired_size = 40
    old_size = img.shape[:2]  # old_size is in (height, width) format

    # taking the ration to the original size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=color)

    return new_img


def preprocessed(data_path):
    """
    in this function we take the images file path test or train images both
    returend as lists after preprocessing and for each image a label is added
    """
    # letter_file_names = next(os.walk(data_path))[1].sort()
    final_imgs = []
    y = []  # making a label from 0 to 26 for each letter
    images = []  # each letter in an array
    for folder_name in tqdm(labels, total=len(labels)):

        folder_path = data_path + folder_name + "/"
        images_in_folder = next(os.walk(folder_path))[2]  # name of each image is returened in a list

        for img_name in images_in_folder:
            image = cv2.imread(folder_path + img_name, 0)
            result = image_processing(image)
            result = hog(result, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False,
                         block_norm="L2")
            images.append(result)
            y.append(folder_name)

    return np.asarray(images), np.asarray(y).astype(np.float32)


def pre_processing(path):
    """
    In this function we read the data from the hdd_dataset and then we read all the images from each file
    in the train folder and test folder and then grayscale then padding them to 256 and then resizing them to
    40X40
    :return:test images and train images labeled
    """

    # Letters are labeled from 0 to 26
    s_time = time()
    print("pre processing images")
    train_imgs, y = preprocessed(path)
    m, s = divmod(calculate_time(s_time), 60)
    print(f" --- done pre processing  images in {int(m):02d}:{int(s):02d}  --- ")
    return train_imgs, y


def chi_square(p, q):
    return 0.5 * np.sum((p - q) ** 2 / (p + q + 0.00001))


def testing(model, k, USE_CHI):
    """
    we take each image of a letter in a specific class
    and predict the label of each image and then calculate the accuracy
    by comparing it to the ground truth label and writing this to a text file
    and by the end we build a confusion matrix and then save it to a csv file
    :param k: number of neighbors
    :param USE_CHI: if used the chi square function or not
    """
    s_time = time()
    print(" Testing The trained model and calculating the accuracy for each class")
    y_test = []
    pred_y = []
    with open("results.txt", "w") as file1:
        if USE_CHI:
            metric = 'Chi Square'
        else:
            metric = 'Euclidean'
        file1.write("k = {0} , distance function is = {1} (L2)\n".format(str(k), metric))
        file1.write("Class Accuracy\n")
        for folder_name in tqdm(labels, total=len(labels)):

            folder_path = testing_path + folder_name + "/"
            images_in_folder = next(os.walk(folder_path))[2]  # name of each image is returened in a list
            images_per_class = []
            y_true = []
            for img_name in images_in_folder:
                image = cv2.imread(folder_path + img_name, 0)
                result = image_processing(image)
                result = hog(result, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             transform_sqrt=False, block_norm="L2")
                images_per_class.append(result)
                y_true.append(int(folder_name))
                y_test.append(int(folder_name))

            pred_per_class = model.predict(np.asarray(images_per_class))
            pred_y.append(pred_per_class)

            acc = accuracy_score(np.asarray(y_true), pred_per_class)

            file1.write("{0}     {1:.2f}%\n".format(folder_name, acc * 100))
    print(" the text file containing the results has been saved")
    m, s = divmod(calculate_time(s_time), 60)
    print(f" --- done testing in {int(m):02d}:{int(s):02d}  --- ")
    return np.asarray(y_test), np.asarray(pred_y, dtype=object)


def train_n_test(k, USE_CHI=1):
    """
    this function trains and tests a knn model for hebrew letters in order to classify each letter
    using either the euclidean distance metric or the Chi squared
    """
    s_time = time()
    if USE_CHI:
        metric = chi_square
    else:
        metric = 'euclidean'

    x, y = pre_processing(training_path)

    print("Training and Fitting the K-NN model")
    train_time = time()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
    model = KNeighborsClassifier(n_neighbors=k,
                                 n_jobs=-1
                                 , metric=metric)
    model.fit(x_train, y_train)

    acc = model.score(x_val, y_val)
    m, s = divmod(calculate_time(train_time), 60)
    print(f" --- the Model is done training in {int(m):02d}:{int(s):02d}  --- ")
    print("The accuracy of the K-nn Model: {:.2f}%".format(acc * 100))

    y_true, pred_y = testing(model, k, USE_CHI)

    print(" Writing confusion_matrix to CSV file")
    pred_y = np.concatenate(pred_y)
    conf_mat = confusion_matrix(y_true, np.reshape(pred_y, (len(y_true),)))
    df = pd.DataFrame(conf_mat)
    df.to_csv("confusion_matrix.csv")
    print(" CSV has been saved to working directory")

    m, s = divmod(calculate_time(s_time), 60)
    print(f" --- done training and testing in {int(m):02d}:{int(s):02d}  --- ")


if __name__ == "__main__":
    if args.metric and args.k:
        train_n_test(int(args.k), USE_CHI=int(args.metric))
    else:
        train_n_test(7)
