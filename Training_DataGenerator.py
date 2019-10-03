"""

class Preprocess
    Calculate possible tile sizes, calculate tile positions, split images based on these calculations.

class Augment
    Read train and label images separately and merge them
    Using Keras preprocessing to augment the merged image
    Separate augmented image back into single train and label image

class Create_npy_files
    ...

class dataProcess

    Create train and test data
    Load train and test data

"""


import numpy as np
import math
import os
import copy
import glob
import cv2
from skimage.measure import label as set_labels, regionprops
from scipy.ndimage.morphology import distance_transform_edt as get_dmap
from keras.preprocessing.image import ImageDataGenerator


class Preprocess:

    def __init__(self, train_path="train" + os.sep + "image", label_path="train" + os.sep + "label",
                 raw_path = "train" + os.sep + "RawImgs", img_type="tif"):

        """
        Using glob to get all .img_type form path
        """

        self.train_imgs = glob.glob(train_path + "" + os.sep + "*." + img_type)
        self.label_imgs = glob.glob(label_path + "" + os.sep + "*." + img_type)
        self.train_path = train_path
        self.raw_path = raw_path
        self.label_path = label_path


    def poss_tile_sizes(self, path):

        """
        get corresponding tile sizes and number of tiles per raw image
        """

        path_raw = path

        for img in os.listdir(path_raw + os.sep + "image"):
            read_img = cv2.imread(path_raw + os.sep + "image" + os.sep + img, -1)
            y,x = read_img.shape

            break

        size = 16

        displ_values = set()
        real_values = []

        print("\n")
        while size < max([y, x]) / 2 + 16:

            x_tile = math.ceil(x / size)
            y_tile = math.ceil(y / size)

            if x_tile > 1 and y_tile > 1:

                x_overlap = (np.abs(x - x_tile * size)) / (x_tile - 1)
                y_overlap = (np.abs(y - y_tile * size)) / (y_tile - 1)

            if (x_overlap.is_integer() and y_overlap.is_integer()) and (x_tile * y_tile) % 2 == 0:

                displ_values.add("Tile size (px): " + str(size) + " | Number of tiles: " + str(x_tile * y_tile))
                real_values.append((size, x_tile * y_tile))

            size += 16

        if x == y and x%16 == 0:
            displ_values.add("Tile size (px): " + str(x) + " | Number of tiles: " + str(1))
            real_values.append((x, 1))


        # using one tile when image size is small but not of square shape and x,y % 16 != 0
        #######

        x_old = x
        x_new = x
        if x>y:
            while x_new%16 != 0:
                x_new+=2

            bs_x = x_new-x_old

        y_new = y
        if x < y:
            while y_new % 16 != 0:
                y_new += 2

            bs_x = y_new - x


        displ_values.add("Tile size (px): " + str(x+bs_x) + " | Number of tiles: " + str(1))
        real_values.append((x+bs_x, 1))


        #######

        return displ_values, real_values


    def find_tile_pos(self, x, y, tile_size, start_x, end_x, start_y, end_y, column, row):

        x_tile = math.ceil(x / tile_size)
        y_tile = math.ceil(y / tile_size)

        #if x_tile > 1 and y_tile > 1:
        if x_tile > 1 or y_tile > 1:

            x_overlap = (np.abs(x - x_tile * tile_size)) / (x_tile - 1)

        if y_tile > 1:

            y_overlap = (np.abs(y - y_tile * tile_size)) / (y_tile - 1)

        # if column greater equal 1 then set start_x and end_x as follows
        if column >= 1:
            start_x = int(column * tile_size - column * x_overlap)
            end_x = int(start_x + tile_size)

        # if row greater equal 1 then set start_y and end_y as follows
        if row >= 1:
            start_y = int((row) * tile_size - (row) * y_overlap)
            end_y = int(start_y + tile_size)

        # if column is equal to number of x tiles, reset start_x, end_x and column (moving to next row)
        if column == x_tile:
            start_x = 0
            end_x = tile_size

            column = 0

        # if column greater equal number of x tiles -1, add 1 to row (moving to next column)
        if column >= x_tile - 1 and row < y_tile - 1:
            row += 1

        column += 1

        return start_x, end_x, start_y, end_y, column, row


    def splitImgs(self, path, tile_size, n_tiles):

        if n_tiles%2!=0 and n_tiles!=1 or tile_size%16!=0:

            print(n_tiles)
            print("Incorrect number of tiles or tile size not divisible by 16.\nAborting")
            exit()

        path_train = path + os.sep + self.train_path
        path_label = path + os.sep + self.label_path
        path_raw = path + os.sep  + self.raw_path

        for img in os.listdir(path_raw + os.sep + "image"):

            read_img = cv2.imread(path_raw + os.sep + "image" + os.sep + img, -1)

            if np.sum(read_img) == 0:
                print("Problem with reading image.\nAborting")
                exit()

            elif np.max(read_img) > 255:
                print("Image bit depth is 16 or higher. Please convert images to 8-bit first.\nAborting")
                exit()

            read_lab = cv2.imread(path_raw + os.sep + "label" + os.sep + img, cv2.IMREAD_GRAYSCALE)

            y, x = read_img.shape

            # todo
            #if tile_size > max(y,x)/2+16 and n_tiles!=1:

            if tile_size > x:

                bs_x = int((tile_size - x) / 2)
                bs_y = 0

                read_img = cv2.copyMakeBorder(read_img, bs_y, bs_y, bs_x, bs_x, cv2.BORDER_REFLECT)
                read_lab = cv2.copyMakeBorder(read_lab, bs_y, bs_y, bs_x, bs_x, cv2.BORDER_REFLECT)

            if tile_size > y:

                bs_y = int((tile_size - y) / 2)
                bs_x = 0

                read_img = cv2.copyMakeBorder(read_img, bs_y, bs_y, bs_x, bs_x, cv2.BORDER_REFLECT)
                read_lab = cv2.copyMakeBorder(read_lab, bs_y, bs_y, bs_x, bs_x, cv2.BORDER_REFLECT)

            # splitting image into n tiles of predefined size
            #############

            # resetting n_tiles based on new image size
            n_tiles = int(math.ceil(y / tile_size) * math.ceil(x / tile_size))

            start_y = 0
            start_x = 0
            end_y = tile_size
            end_x = tile_size

            column = 0
            row = 0

            for i in range(n_tiles):

                start_x, end_x, start_y, end_y, column, row = self.find_tile_pos(x, y, tile_size, start_x, end_x, start_y, end_y,
                                                                    column, row)

                image_tile_train = read_img[start_y:end_y, start_x:end_x]
                image_tile_label = read_lab[start_y:end_y, start_x:end_x]

                cv2.imwrite(path_train + os.sep  + str(i) + "_" + img, image_tile_train)
                cv2.imwrite(path_label + os.sep  + str(i) + "_" + img, image_tile_label)

                #############


class Augment:

    def __init__(self, path, shear_range, rotation_range, zoom_range, brightness_range, horizontal_flip, vertical_flip,
                 width_shift_range, height_shift_range, train_path="train" + os.sep + "image", label_path="train" + os.sep + "label",
                 raw_path = "train" + os.sep + "RawImgs", merge_path="merge", aug_merge_path="aug_merge",
                 aug_train_path="aug_train", aug_label_path="aug_label", img_type="tif", weights_path="weights",
                 aug_weights_path="aug_weights"):

        """
        Using glob to get all .img_type form path
        """

        self.path = path
        self.shear_range = shear_range
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

        self.train_imgs = glob.glob(self.path + os.sep + train_path + os.sep + "*." + img_type)
        self.label_imgs = glob.glob(self.path + os.sep + label_path + os.sep + "*." + img_type)
        self.train_path = self.path + os.sep + train_path
        self.raw_path = self.path + os.sep + raw_path
        self.label_path = self.path + os.sep + label_path
        self.merge_path = self.path + os.sep + merge_path
        self.img_type = img_type
        self.aug_merge_path = self.path + os.sep + aug_merge_path
        self.aug_train_path = self.path + os.sep + aug_train_path
        self.aug_weights_path = self.path + os.sep + aug_weights_path
        self.aug_label_path = self.path + os.sep + aug_label_path
        self.slices = len(self.train_imgs)

        self.map_path = self.path + os.sep + weights_path


        # ImageDataGenerator performs augmentation on original images
        self.datagen = ImageDataGenerator(

            shear_range=self.shear_range,
            rotation_range=self.rotation_range,
            zoom_range=self.zoom_range,
            brightness_range=self.brightness_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            fill_mode='reflect')  # pixels outside boundary are set to 0

    def start_augmentation(self, imgnum, wmap, tile_size):

        def create_distance_weight_map(label, w0=10, sigma=5):

            # creating first parameter of weight map formula
            template_weight_map = np.ones_like(label)
            template_weight_map[label > 0] = 2

            # setting all 255 values to 1
            label[label > 1] = 1
            # inverting label for distance_transform
            new_label = 1 - label

            # calculate distance_transform
            dist_map1 = get_dmap(new_label)

            # labels each separable object with one unique pixel value
            labelled = set_labels(label)

            # creates list with label properties (for us important: coordinates)
            regprops = regionprops(labelled)

            stack = []

            # iterate through every object in image
            for i in regprops:

                # create shallow copy of new_label (modifying matrix, without changing original)
                temp = copy.copy(new_label)

                # iterate through coordinates of each object
                for n in i.coords:
                    # create one image each, in which one object is removed (background = 1)
                    temp[n[0], n[1]] = 1

                stack.append(get_dmap(temp))

            # create empty matrix
            dist_map2 = np.zeros_like(label)

            x = 0
            # iterate through each row of distance map 1
            for row in dist_map1:

                y = 0
                # iterate through each column
                for col in row:
                    for img in stack:

                        # check if at position x,y the pixel value of img is bigger than dist_map1 >> distance to second nearest border
                        if img[x, y] > dist_map1[x, y]:

                            dist_map2[x, y] = img[x, y]
                            break

                        else:
                            dist_map2[x, y] = dist_map1[x, y]

                    y += 1
                x += 1

            weight_map = template_weight_map + w0 * np.exp(- ((dist_map1 + dist_map2) ** 2 / (2 * sigma ** 2)))

            return weight_map

        print("Starting Augmentation \n")

        """
        Start augmentation.....
        """

        trains =  self.train_imgs
        labels =  self.label_imgs
        path_train =  self.train_path
        path_label =  self.label_path
        path_merge =  self.merge_path
        path_aug_merge = self.aug_merge_path

        # checks if number of files in train and label folder are equal
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("Number of train images does match number of label images.\nAborting")
            exit()

        # iterate through folder, merge label, original images and save to merged folder
        for count, image in enumerate(os.listdir(path_train)):

            print(image)

            x_t = cv2.imread(path_train + os.sep  + image, cv2.IMREAD_GRAYSCALE)
            x_l = cv2.imread(path_label + os.sep  + image, cv2.IMREAD_GRAYSCALE)

            # exclude image tiles without any labels
            if np.count_nonzero(x_l) == 0:
                pass

            else:

                if x_t.shape[1] < tile_size or x_t.shape[0] < tile_size:
                    bs_x = int((tile_size - x_t.shape[1]) / 2)
                    bs_y = int((tile_size - x_t.shape[0]) / 2)

                    x_t = cv2.copyMakeBorder(x_t, bs_y, bs_y, bs_x, bs_x, cv2.BORDER_REFLECT)
                    x_l = cv2.copyMakeBorder(x_l, bs_y, bs_y, bs_x, bs_x, cv2.BORDER_REFLECT)

                if wmap == False:
                    x_w = np.zeros((x_l.shape[0], x_l.shape[1]))

                else:
                    #create weight map
                    x_w = create_distance_weight_map(x_l)

                # create empty array (only 0s) with shape (x,y, number of channels)
                aug_img = np.zeros((x_t.shape[0], x_l.shape[1], 3))

                # setting each channel to label, empty array and original

                aug_img[:, :, 2] = x_l
                aug_img[:, :, 1] = x_w
                aug_img[:, :, 0] = x_t

                if wmap == True:

                    #increasing intensity values of label images (to 255 if value was > 0)
                    for x in np.nditer(aug_img[:,:,2], op_flags=['readwrite']):
                        x[...] = x * 255


                # write final merged image
                aug_img = aug_img.astype('uint8')
                cv2.imwrite(path_merge + os.sep + image, aug_img)

                img = aug_img
                img = img.reshape((1,) + img.shape)

                savedir = path_aug_merge + os.sep + image

                if not os.path.lexists(savedir):
                    os.mkdir(savedir)

                self.doAugmentate(img, savedir, image, imgnum)


        aug_params_file = open(self.path + os.sep + "augmentation_parameters.txt", "w")
        aug_params_file.write("Horizontal flip: " + str(self.horizontal_flip) +
                              "\nVertical flip: " + str(self.vertical_flip) +
                              "\nWidth shift range: " + str(self.width_shift_range) +
                              "\nHeight shift range: " + str(self.height_shift_range) +
                              "\nShear range: " + str(self.shear_range) +
                              "\nRotation range: " + str(self.rotation_range) +
                              "\nZoom range: " + str(self.zoom_range) +
                              "\nBrightness range: " + str(self.brightness_range))

        aug_params_file.close()


    def doAugmentate(self, img, save_to_dir, save_prefix, imgnum , batch_size=1, save_format='tif'):

        """
        augment images
        """
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):
            i += 1

            if i >= imgnum:
                break

    def splitMerge(self, wmap):

        print("Splitting merged images")

        """
        split merged image apart
        """

        path_merge =  self.aug_merge_path
        path_train =  self.aug_train_path
        path_weights =  self.aug_weights_path
        path_label =  self.aug_label_path

        print(path_train)

        for image in os.listdir(path_merge):

            path = path_merge + os.sep  + image

            train_imgs = glob.glob(path + os.sep + "*." + self.img_type)


            def save_dir(path):
                savedir = path + os.sep  + image
                if not os.path.lexists(savedir):
                    os.mkdir(savedir)

            save_dir(path_train)
            save_dir(path_label)

            if wmap == True:
                save_dir(path_weights)

            for imgname in train_imgs:

                midname = imgname.split(os.sep)[-1]
                img = cv2.imread(imgname)

                print(midname)

                img_train = img[:, :, 2]  # cv2 read image rgb->bgr
                img_label = img[:, :, 0]

                # decided to keep varying intensity values at border as this seems to increase segm-performance
                # setting intensity values back to 255 after brightness change
                #img_label[img_label > 0] = 255

                cv2.imwrite(path_train + os.sep + image + os.sep + midname, img_train)
                cv2.imwrite(path_label + os.sep + image + os.sep + midname, img_label)

                if wmap==True:
                    img_weights = img[:, :, 1]
                    cv2.imwrite(path_weights + os.sep + image + os.sep + midname, img_weights)

        print("\nsplitMerge finished")


class Create_npy_files(Preprocess):

    def __init__(self, path, data_path="aug_train", label_path="aug_label", weight_path="aug_weights",
                 npy_path="npydata", img_type="tif"):

        Preprocess.__init__(self, train_path="train" + os.sep + "image", label_path="train" + os.sep + "label",
                            raw_path = "train" + os.sep + "RawImgs", img_type=img_type)

        self.path = path
        self.data_path = self.path + os.sep + data_path
        self.label_path = self.path + os.sep + label_path
        self.img_type = img_type
        self.npy_path = self.path + os.sep + npy_path
        self.weight_path = self.path + os.sep + weight_path


    def create_train_data(self, wmap, out_rows, out_cols):

        """
        adding all image data to one numpy array file (npy)

        all mask image files are added to imgs_mask_train.npy
        all original image files are added to imgs_train.npy

        all weight image files are added to weight_train.npy
        """

        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        imgs = glob.glob(self.data_path + os.sep + "*" + os.sep + "*")
        print(imgs)

        imgdatas = np.ndarray((len(imgs), out_rows, out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), out_rows, out_cols, 1), dtype=np.uint8)
        imgweights = np.ndarray((len(imgs), out_rows, out_cols, 1), dtype=np.uint8)

        width = out_cols
        height = out_rows

        for imgname in imgs:

            midname = imgname.split(os.sep )[-2] + os.sep  + imgname.split(os.sep )[-1]

            print(self.data_path + os.sep  + midname)

            img = cv2.imread(self.data_path + os.sep  + midname, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(self.label_path + os.sep  + midname, cv2.IMREAD_GRAYSCALE)

            img = np.array([img])
            img = img.reshape((width, height, 1))

            label = np.array([label])
            label = label.reshape((width, height, 1))

            imgdatas[i] = img
            imglabels[i] = label

            if wmap==True:

                weights = cv2.imread(self.weight_path + os.sep  + midname,cv2.IMREAD_GRAYSCALE)

                weights = np.array([weights])
                weights = weights.reshape((width, height, 1))

                imgweights[i] = weights

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('Loading done')

        # original
        np.save(self.npy_path + os.sep + 'imgs_train.npy', imgdatas)
        np.save(self.npy_path + os.sep + 'imgs_mask_train.npy', imglabels)

        if wmap==True:
            np.save(self.npy_path + os.sep + 'imgs_weights.npy', imgweights)

        print('Saving to .npy files done.')

    def check_class_balance(self):

        label_array = np.load(self.npy_path + os.sep + "imgs_mask_train.npy")

        tile_size = label_array[0].shape[0]

        l = []
        for count, i in enumerate(label_array):

            b = len(i[i == 0])
            l.append(b / (tile_size ** 2))

        av = np.average(l)

        return av, round(1/(1-av), 0)




