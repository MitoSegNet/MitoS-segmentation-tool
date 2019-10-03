""""

class GPU or CPU
    Lets user decide which component to use for processing

class MitoSegNet
    Main Deep Learning Architecture (based on U-Net)

    number of conv layers: 24
    number of relu units: 23
    number of sigmoid units: 1 (after last conv layer)
    number of batch norm layers: 10
    number of max pooling layers: 4

"""


import os
import cv2
import numpy as np
import pandas as pd
import math
import copy
import glob
from time import time
from math import sqrt
from skimage.morphology import remove_small_objects
from scipy.ndimage import label
from screeninfo import get_monitors
from tkinter import *


class GPU_or_CPU:

    def __init__(self, mode):

        self.mode = mode

    def ret_mode(self):

        if self.mode == "GPU":
            print("Train / Predict on GPU")

        elif self.mode == "CPU":
            print("Train / Predict on CPU")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        return self.mode


from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal as gauss
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf

# ignoring deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Training_DataGenerator import *


class MitoSegNet:

    def __init__(self, path, img_rows, img_cols, org_img_rows, org_img_cols):

        self.path = path

        self.img_rows = img_rows
        self.img_cols = img_cols

        self.org_img_rows = org_img_rows
        self.org_img_cols = org_img_cols

    def natural_keys(self, text):

        # sorting of alphanumerical strings

        def atoi(text):
            return int(text) if text.isdigit() else text

        return [atoi(c) for c in re.split('(\d+)', text)]

    def load_data(self, wmap, vbal):

        print('-' * 30)
        print('Load train images...')
        print('-' * 30)

        imgs_train = np.load(self.path + os.sep + "npydata" + os.sep +"imgs_train.npy")
        imgs_mask_train = np.load(self.path + os.sep + "npydata" + os.sep + "imgs_mask_train.npy")


        """
        # checking label data for values other than 0 or 255
        l_int = list(range(1, 255))
        print("Checking label data")
        l_check = (np.isin(np.unique(imgs_mask_train), l_int))

        if True in l_check:
            
            print("Errors in binary mask detected. Aborting.")
            exit()
        """

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        imgs_train /= 255
        imgs_mask_train /= 255

        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0

        if wmap == True:

            imgs_weights = np.load(self.path + os.sep + "npydata" + os.sep + "imgs_weights.npy")
            imgs_weights = imgs_weights.astype('float32')

            # due to brightness changes in augmentation, some weight images have a 0 background
            imgs_weights[imgs_weights == 0] = 1

            # setting background pixel weights to vbal (because of class imbalance)
            imgs_weights[imgs_weights == 1] = vbal

            return imgs_train, imgs_mask_train, imgs_weights

        else:

            return imgs_train, imgs_mask_train

    def get_mitosegnet(self, wmap, lr):

        inputs = Input(shape=(self.img_rows, self.img_cols, 1))
        print(inputs.get_shape(), type(inputs))


        # core mitosegnet (modified u-net) architecture
        # batchnorm architecture (batchnorm before activation)
        ######################################################################

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer=gauss())(inputs)
        print("conv1 shape:", conv1.shape)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation("relu")(batch1)

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(act1)  # conv1
        print("conv1 shape:", conv1.shape)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation("relu")(batch1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(act1)
        print("pool1 shape:", pool1.shape)
        ########

        ########
        conv2 = Conv2D(128, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(pool1)
        print("conv2 shape:", conv2.shape)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation("relu")(batch2)

        conv2 = Conv2D(128, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(act2)  # conv2
        print("conv2 shape:", conv2.shape)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation("relu")(batch2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(act2)
        print("pool2 shape:", pool2.shape)
        ########

        ########
        conv3 = Conv2D(256, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(pool2)
        print("conv3 shape:", conv3.shape)
        batch3 = BatchNormalization()(conv3)
        act3 = Activation("relu")(batch3)

        conv3 = Conv2D(256, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(act3)  # conv3
        print("conv3 shape:", conv3.shape)
        batch3 = BatchNormalization()(conv3)
        act3 = Activation("relu")(batch3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(act3)
        print("pool3 shape:", pool3.shape)
        ########

        ########
        conv4 = Conv2D(512, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(pool3)
        batch4 = BatchNormalization()(conv4)
        act4 = Activation("relu")(batch4)

        conv4 = Conv2D(512, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(act4)  # conv4
        batch4 = BatchNormalization()(conv4)
        act4 = Activation("relu")(batch4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(act4)
        ########

        ########
        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(pool4)
        batch5 = BatchNormalization()(conv5)
        act5 = Activation("relu")(batch5)

        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(act5)  # conv5
        batch5 = BatchNormalization()(conv5)
        act5 = Activation("relu")(batch5)

        ########

        up6 = Conv2D(512, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(UpSampling2D(size=(2, 2))(act5))

        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(conv6)


        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(conv7)


        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(conv8)


        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(UpSampling2D(size=(2, 2))(conv8))

        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        ######################################################################

        conv10 = Conv2D(1, 1, activation='sigmoid', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 2))))(conv9)

        if wmap == False:
            input = inputs
            loss = self.pixelwise_crossentropy()
        else:
            weights = Input(shape=(self.img_rows, self.img_cols, 1))
            input = [inputs, weights]

            loss = self.weighted_pixelwise_crossentropy(input[1])

        model = Model(inputs=input, outputs=conv10)

        model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['accuracy', self.dice_coefficient])

        return model

    def dice_coefficient(self, y_true, y_pred):

        smooth = 1

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)

        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return dice

    def pixelwise_crossentropy(self):

        def loss(y_true, y_pred):

            return losses.binary_crossentropy(y_true, y_pred)

        return loss

    def weighted_pixelwise_crossentropy(self, wmap):

        def loss(y_true, y_pred):

            return losses.binary_crossentropy(y_true, y_pred) * wmap

        return loss


    def train(self, epochs, learning_rate, batch_size, wmap, vbal, model_name, new_ex):

        if ".hdf5" in model_name:
            model_name = model_name.split(".hdf5")[0]
        else:
            pass

        print("Loading data")

        if wmap == False:
            imgs_train, imgs_mask_train = self.load_data(wmap=wmap, vbal=vbal)
        else:
            imgs_train, imgs_mask_train, img_weights = self.load_data(wmap=wmap, vbal=vbal)

        print("Loading data done")

        model = self.get_mitosegnet(wmap, learning_rate)
        print("Got MitoSegNet")

        print(self.path + os.sep + model_name)

        if os.path.isfile(self.path + os.sep + model_name + ".hdf5"):

            model.load_weights(self.path + os.sep + model_name + ".hdf5")
            print("Loading weights")

        else:
            print("No previously optimized weights were loaded. Proceeding without")

        # Set network weights saving mode.
        # save previously established network weights (saving model after every epoch)

        print('Fitting model...')

        if new_ex == "New":

            first_ep = 0
            model_name = model_name + "_" + str(self.img_rows) + "_"

        elif new_ex == "Finetuned_New":

            first_ep = 0

        else:
            prev_csv_file = pd.read_csv(self.path + os.sep + model_name + 'training_log.csv')
            first_ep = len(prev_csv_file)

            if prev_csv_file.shape[1] > 7:
                prev_csv_file = prev_csv_file.drop(prev_csv_file.columns[[0]], axis=1)

        csv_logger = CSVLogger(self.path + os.sep + model_name + 'training_log.csv')

        tensorboard = TensorBoard(log_dir=self.path + os.sep + "logs/{}".format(time()))

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                     ModelCheckpoint(filepath=self.path + os.sep + model_name + ".hdf5",
                                     monitor='val_loss', verbose=1, save_best_only=True),
                     csv_logger, tensorboard]

        if wmap == True:
            x = [imgs_train, img_weights]
        else:
            x = imgs_train

        print("\nCopy the line below into the terminal, press enter and click on the link to evaluate the training "
              "performance:\n\ntensorboard --logdir=" + self.path + os.sep + "logs/\n")

        model.fit(x=x, y=imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_split=0.2, shuffle=True, callbacks=callbacks)

        csv_file = pd.read_csv(self.path + os.sep + model_name + 'training_log.csv')

        if new_ex == "New" or new_ex == "Finetuned_New":

            csv_file["epoch"] = list(range(1, len(csv_file) + 1))
            last_ep = len(csv_file)

        if new_ex == "Existing":

            frames = [prev_csv_file, csv_file]
            merged = pd.concat(frames, names=[])

            merged["epoch"] = list(range(1, len(merged) + 1))

            last_ep = len(merged)

            merged.to_csv(self.path + os.sep + model_name + 'training_log.csv')


        info_file = open(self.path + os.sep + model_name + str(first_ep) + "-" + str(last_ep) + "_train_info.txt", "w")
        info_file.write("Learning rate: " + str(learning_rate)+
                        "\nBatch size: " + str(batch_size)+
                        "\nClass balance weight factor: " + str(vbal))
        info_file.close()

        K.clear_session()


    def predict(self, test_path, wmap, tile_size, model_name, pretrain, min_obj_size, ps_filter):

        K.clear_session()

        org_img_rows = self.org_img_rows
        org_img_cols = self.org_img_cols

        natural_keys = self.natural_keys

        def create_test_data(tile_size):

            # adding all image data to one numpy array file (npy)
            # all original image files are added to imgs_test.npy

            i = 0
            print('-' * 30)
            print('Creating test images...')
            print('-' * 30)

            imgs = glob.glob(test_path + os.sep + "*")

            # adding a border around image to avoid using segmented border regions for final mask
            if org_img_cols < tile_size:
                bs_x = int((tile_size - org_img_cols) / 2)
            else:
                bs_x = 40

            if org_img_rows < tile_size:
                bs_y = int((tile_size - org_img_rows) / 2)
            else:
                bs_y = 40


            def get_tile_values(org_img_cols, org_img_rows, bs_x, bs_y, tile_size):

                x = org_img_cols + 2 * bs_x
                y = org_img_rows + 2 * bs_y

                x_tile = math.ceil(x / tile_size)
                y_tile = math.ceil(y / tile_size)

                x_overlap = (np.abs(x - x_tile * tile_size)) / (x_tile - 1)
                y_overlap = (np.abs(y - y_tile * tile_size)) / (y_tile - 1)

                return x, y, x_tile, y_tile, x_overlap, y_overlap

            x, y, x_tile, y_tile, x_overlap, y_overlap = get_tile_values(org_img_cols, org_img_rows, bs_x, bs_y,
                                                                         tile_size)

            while not x_overlap.is_integer():

                bs_x+=1
                x, y, x_tile, y_tile, x_overlap, y_overlap = get_tile_values(org_img_cols, org_img_rows, bs_x, bs_y,
                                                                             tile_size)
    
            while not y_overlap.is_integer():
                bs_y+=1
                x, y, x_tile, y_tile, x_overlap, y_overlap = get_tile_values(org_img_cols, org_img_rows, bs_x, bs_y,
                                                                             tile_size)

            n_tiles = x_tile * y_tile
            ###############

            # added 05/12/18 to avoid underscores causing problems when stitching images back together
            # if any("_" in s for s in imgs):
            for img in imgs:

                if "_" in img and ".tif" in img:

                    # split file path without filename
                    img_edited = img.split(os.sep)[:-1]

                    # join list back to path string
                    img_edited_path = os.sep.join(img_edited)

                    img_name = img.split(os.sep)[-1]
                    img_name = img_name.replace("_", "-")

                    os.rename(img, img_edited_path + os.sep + img_name)

            imgs = glob.glob(test_path + os.sep + "*.tif")
            imgs.sort(key=natural_keys)

            # create list of images that correspond to arrays in npy file
            ################
            mod_imgs = []
            for x in imgs:

                part = x.split(os.sep)

                c = 0
                while c <= n_tiles - 1:

                    temp_str = os.sep.join(part[:-1])

                    if ".tif" in part[-1]:

                        mod_imgs.append(temp_str + os.sep + str(c) + "_" + part[-1])

                    c += 1


            imgdatas = np.ndarray((len(imgs) * n_tiles, tile_size, tile_size, 1), dtype=np.uint8)

            for imgname in imgs:

                if ".tif" in imgname:

                    print(imgname)

                    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)

                    # adding padding around image to avoid prediction at border

                    # top, bottom, left, right
                    pad_img = cv2.copyMakeBorder(img, bs_y, bs_y, bs_x, bs_x, cv2.BORDER_REFLECT)
                    cop_img = copy.copy(pad_img)

                    y, x = pad_img.shape

                    # split into n tiles

                    start_y = 0
                    start_x = 0

                    end_y = tile_size
                    end_x = tile_size

                    column = 0
                    row = 0

                    for n in range(n_tiles):

                        start_x, end_x, start_y, end_y, column, row = preproc.find_tile_pos(x, y, tile_size, start_x, end_x,
                                                                                         start_y, end_y, column, row)

                        img_tile = cop_img[start_y:end_y, start_x:end_x]

                        img = img_tile.reshape((tile_size, tile_size, 1))

                        imgdatas[i] = img

                        i += 1

                np.save(test_path + os.sep + 'imgs_array.npy', imgdatas)

            return mod_imgs, y, x, bs_x, bs_y, x_tile, y_tile, x_overlap, y_overlap, n_tiles


        def load_test_data():

            print('-' * 30)
            print('Load test images...')
            print('-' * 30)

            imgs_test = np.load(test_path + os.sep + "imgs_array.npy")
            imgs_test = imgs_test.astype('float32')

            imgs_test /= 255

            return imgs_test

        preproc = Preprocess()

        l_imgs, y, x, bs_x, bs_y, x_tile, y_tile, x_overlap, y_overlap, n_tiles = create_test_data(int(tile_size))
        imgs_test = load_test_data()

        # predict if no npy array exists yet
        if not os.path.isfile(test_path + os.sep + "imgs_mask_array.npy"):

            lr = 1e-4

            model = self.get_mitosegnet(wmap, lr)

            if pretrain == "":
                model.load_weights(self.path + os.sep + model_name)
            else:
                model.load_weights(pretrain)

            print('Predict test data')

            imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

            np.save(test_path + os.sep + 'imgs_mask_array.npy', imgs_mask_test)

        else:

            print("\nFound imgs_mask_array.npy. Skipping prediction and converting array to images\n")

        org_img_list = []

        for img in l_imgs:

            # split file path without filename

            if ".tif" in img:

                img_name = img.split(os.sep)[-1]
                img_name = img_name.split("_")[1]

                org_img_list.append(img_name)

        org_img_list = list(set(org_img_list))
        org_img_list.sort(key=self.natural_keys)

        # saving arrays as images
        print("Array to image")

        imgs = np.load(test_path + os.sep + 'imgs_mask_array.npy')

        #####
        imgs = imgs.astype('float32')
        #####

        start_y = 0
        start_x = 0
        end_y = tile_size
        end_x = tile_size

        column = 0
        row = 0

        img_nr = 0
        org_img_list_index = 0

        for n, image in zip(range(imgs.shape[0]), l_imgs):

            if img_nr == 0:

                current_img = np.zeros((org_img_rows, org_img_cols))

            img = imgs[n]

            img = np.array(img)
            img = img.reshape((tile_size, tile_size))

            #############################################################
            # if we are in first or last column, the real x tile size is dependant on both border size and x overlap
            if column == 0 or column == x_tile - 1:

                real_x_tile = int(tile_size - bs_x - x_overlap / 2)
                #border_x = bs_x

            # if we are in first or last row, the real y tile size is dependant on both border size and y overlap
            if row == 0 or row == y_tile - 1:
                real_y_tile = int(tile_size - bs_y - y_overlap / 2)
                #border_y = bs_y

            # first tile of next row
            if column == 0 and row != 0:

                start_y = int(y_overlap / 2)

                final_start_y = final_end_y

                start_x = bs_x
                end_x = int(real_x_tile + bs_x)

                final_start_x = 0
                final_end_x = int(real_x_tile)

                # if we are between the first and last row the real y tile size depends only on the overlap
                if row != 0 and row != y_tile - 1:
                    real_y_tile = int(tile_size - y_overlap)
                    #border_y = 0

                end_y = int(start_y + real_y_tile)

                final_end_y = int(final_start_y + real_y_tile)

            # first tile
            if column == 0 and row == 0:
                start_x = bs_x
                end_x = int(real_x_tile + bs_x)

                start_y = bs_y
                end_y = int(real_y_tile + bs_y)

                final_start_x = 0
                final_end_x = int(real_x_tile)

                final_start_y = 0
                final_end_y = int(real_y_tile)

            column += 1

            # last column tile
            if column == x_tile:
                start_x = int(start_x)

                org_end_x = x - bs_x

                end_x = tile_size - bs_x

                final_end_x = org_end_x

                column = 0
                row += 1

            # iterate over columns

            # prior to stitching the overlapping sections and the padding have to be removed
            cut_img = img[int(start_y):int(end_y), int(start_x):int(end_x)]

            current_img[int(final_start_y):int(final_end_y), int(final_start_x):int(final_end_x)] = cut_img

            start_x = int(x_overlap / 2)

            final_start_x += int(real_x_tile)

            # real tile size is still set to old value
            if column != 0 and column != x_tile - 1:
                real_x_tile = int(tile_size - x_overlap)

            end_x = start_x + int(real_x_tile)

            final_end_x = int(final_start_x + real_x_tile)

            #############################################################

            current_img = current_img.astype(np.float32)

            img_nr += 1

            # once one image has been fully stitched, remove any objects below 10 px size and save
            if img_nr == n_tiles:

                column = 0
                row = 0

                # convert to binary
                current_img[current_img > 0.5] = 255
                current_img[current_img <= 0.5] = 0

                current_img = current_img.astype(np.uint8)

                label_image, num_features = label(current_img)

                # allow user to specify what minimum object size should be (originally set to 10)
                new_image = remove_small_objects(label_image, int(min_obj_size))

                new_image[new_image != 0] = 255

                """       
                display both the prediction and the original image and allow the user
                to select which images will be saved and which ones discarded    
                """

                ############################################

                if ps_filter == "1":

                    screen_res = str(get_monitors()[0])
                    screen_res = (screen_res.split("(")[1])

                    x_res = int(screen_res.split("x")[0])

                    y_res = screen_res.split("x")[1]
                    y_res = int(y_res.split("+")[0])


                    cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)
                    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

                    # image resizing factor for display
                    f = 2.5 * (x / x_res)

                    # x , y
                    cv2.resizeWindow('Prediction', int(x / f), int(y / f))
                    cv2.moveWindow("Prediction", int(0.1 * x_res), int(0.1 * y_res))

                    cv2.resizeWindow('Original', int(x / f), int(y / f))
                    cv2.moveWindow("Original", int(0.5 * x_res), int(0.1 * y_res))

                    cv2.imshow("Prediction", new_image.astype(np.uint8))

                    org_img = cv2.imread(test_path + os.sep + org_img_list[org_img_list_index])
                    cv2.imshow("Original", org_img)

                    print("\nPress s to save the image and any other key to discard it\n")

                    cv2.waitKey(115)

                    if cv2.waitKey() == ord("s"):
                        print(org_img_list[org_img_list_index] + " saved")
                        cv2.imwrite(test_path + os.sep + "Prediction" + os.sep + org_img_list[org_img_list_index], new_image)

                    else:
                        print(org_img_list[org_img_list_index] + " discarded")

                    cv2.destroyAllWindows()

                else:
                    cv2.imwrite(test_path + os.sep + "Prediction" + os.sep + org_img_list[org_img_list_index], new_image)

                org_img_list_index+=1
                img_nr = 0

        K.clear_session()
