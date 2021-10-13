# NAME
#  pic_laoder
#
# HELP
#   usually not called directly (no main call)
#
# DESCRIPTION
#    Reads images and their additional information
#    Does data modification as auto contrast or image roation
#
# COPYRIGHT
#   Mycronic AB 2017.
#   THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF MYCRONIC AB.
#
# AUTHOR
#   Eric Kolibacz, eric.kolibacz@mycronic.com
#

# Libraries
# Standard library
import sys
import os
import pickle
import datetime
import copy
import time

# Third-party libraries
import numpy as np
from PIL import Image, ImageOps
import psycopg2
from io import BytesIO
import requests

# Own libraries
from add_ons import split_set


# Defines a neural network - the only function so far is 'training'
class DataLoader(object):
    def __init__(self, specs):

        # All prior defined specifications are now saved to the pic_loader class
        # all given information which are important for the reading process
        self.data_amount = specs["data_amount"]  # how many images
        self.data_dir = specs["data_dir"]  # what is the parent directory to read the files from

        self.components = specs["components"]  # which components to read (folders in parent dir)
        self.components_ratio = specs["components_ratio"]  # how many images how which component
        self.classes = specs["classes"]  # which classes to read from
        self.class_ratio = specs["classes_ratio"]  # how many images from which classes

        self.priority = specs["data_priority"]  # which priority is set for reading data
        self.augment_data = specs["augment_data"]  # Shall training data set be augmented ?

        self.width = specs["dimension"][0]  # width of the image in pixels
        self.height = specs["dimension"][0]  # height of the image in pixels
        self.cropping = specs["cropping"] / float(self.width)  # how many pixels to crop on all side of an image

        self.validation_size = specs['validation_size']  # size of the validation set in percent from data_amount
        self.test_size = specs["test_size"]  # size of the test set in percent from data_amount

        self.add_info = specs["add_info"]  # which additional information to add

        # list containing all the information
        self.training_set = {"features": [], "label": [], "id": [], "add_info": []}
        self.validation_set = {"features": [], "label": [], "id": [], "add_info": []}
        self.testing_set = {"features": [], "label": [], "id": [], "add_info": []}

        # these are the information given in new_data.dat file - they are spereated by ',' are in corresponding order
        self.info_list = ["name", "id", "angle", "error_tag", "bodywidth", "bodylength", "compheight",
                     "ambient", "front", "darkfield", "compdimsamples", "toolname"]
        # these are the maximums for each of the information
        # surely name, id, angle (special case) and label do not have a max and min
        # these others (except tool) have one and need them so that the used values are initialized
        self.max_min_list = [(None, None), (None, None), (None, None), (None, None), (5000, 0),
                        (5000, 0), (5000, 0), (100, 0), (100, 0), (100, 0), (1, 0)]
        # each tool is assigned a value so that when fed into the network a vector is used (see one-hot conversion)
        self.tools = {'A14': 1, 'H09': 2, 'H08': 3, 'C24': 4, 'A13': 5, 'A12': 6, 'H10': 7, 'H02': 8, 'H01': 9, 'H13': 10,
                 'H07': 11, 'H06': 12, 'H05': 13, 'H04': 14, 'D12': 15, 'H03': 16, 'B14': 17, 'A23': 18, 'H11': 19,
                 'H12': 20, 'C14': 21}
        # this implements one hot conversion, 1 corresponds to a 21 entry vector with 1 at 1st entry and 0 otherwise
        # 2 is a 21 vector with 1 at 2nd entry and 0 otherwise, and so on
        total_keys = len(self.tools.keys())
        for key in self.tools.keys():
            vector = [0. for _ in range(total_keys)]
            vector[self.tools[key]-1] = 1.
            self.tools[key] = vector

        # if all information are required (often the case) then the add_info will be ['all'] instead of a list
        # containing the used information - then the list will be created here automatically
        self.add_info_indices = []
        if self.add_info == ["all"]:
            for i in range(4, len(self.info_list)):
                self.add_info_indices.append(i)
        else:   # otherwise the list contains the defined information
            for item in self.add_info:
                self.add_info_indices.append(self.info_list.index(item))

    # the loader function - requires specifications which contains information of how to read which data
    def load_data(self):
        ping = datetime.datetime.now()
        print("Start reading data ...")

        # read data for each component
        for i in range(len(self.components)):
            # run the sample handling function defined below
            tr_s, v_s, te_s = self.sample_handling(
                os.path.expanduser(self.data_dir) + self.components[i], self.components_ratio[i])

            # the extracted information are added to the global dictionaries
            for key in self.training_set:
                self.training_set[key] += tr_s[key]
                self.validation_set[key] += v_s[key]
                self.testing_set[key] += te_s[key]

            # recalculating the ratios for the components (ensures that data amount remains as defined)
            # more information given at the function
            self.components_ratio[i] = self.calc_data_amount(tr_s, v_s, te_s)
            if self.priority != "ratio" and self.priority != "components_ratio" and self.components_ratio[i+1:]:
                self.components_ratio[i+1:] = \
                    self.adjust_ratio(self.components_ratio[i+1:],
                                      self.data_amount - sum(self.components_ratio[:i+1]))

        # calculating the reading speed of the process
        num_data = self.calc_data_amount(self.training_set, self.validation_set, self.testing_set)
        delta_t = (datetime.datetime.now() - ping)
        delta_t = delta_t.seconds + delta_t.microseconds / 1000000.
        print("I pulled " + str(num_data) + " images. It took me " + str(delta_t) + " seconds.")
        print("That's about " + str(num_data / float(delta_t)) + " images per second.")

        print("Training and test data was collected and sorted. \n")

    def load_data_from_query(self, query):
        ping = datetime.datetime.now()

        print("Start reading data ...")

        # read data
        with psycopg2.connect(host="panaxia2.miclaser.net", database="logdb", user="postgres") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT imagename, id, presangle, errortag, "
                            "bodywidth, bodylength, compheight, ambient, front, darkfield, compdimsamples, toolname "
                            "FROM vl LEFT JOIN vlpackage ON vl.idpackage = vlpackage.idpackage "
                            "WHERE " + query)

                information = cur.fetchall()
                beginning = time.time()
                count = 0
                for imagename, img_id, angle, errortag, bodywidth, b_length, compheight, \
                    ambient, front, darkfield, compdimsamples, toolname in information:
                    split = imagename.split('.')
                    img_machine = split[0]
                    img_date = split[2]
                    img_time = split[3]
                    img_hour = img_time[0:2]
                    # fileserver = "http://panaxia2.miclaser.net/log/images/"
                    fileserver = "/mnt/luxornas/images/"
                    file_path = fileserver + img_machine + "/" + img_date + "/" + img_hour + "/" + imagename

                    # response = requests.get(file_path)
                    # if response.status_code == 200:
                    #     img = Image.open(BytesIO(response.content))
                    if os.path.isfile(file_path):
                        img = Image.open(file_path)
                        features = self.data_initialization(img, angle)
                        features = np.array(features)/255.0

                        line = [imagename, img_id, angle, errortag, bodywidth, b_length, compheight,
                                ambient, front, darkfield, compdimsamples, toolname]
                        add_info = self.add_additonal_information(line)

                        self.training_set["features"].append(features)
                        self.training_set["add_info"].append(add_info)
                        self.training_set["id"].append(img_id)

                    self.calc_progress(count, len(information), 0.5)
                    count += 1
        # calculating the reading speed of the process
        num_data = self.calc_data_amount(self.training_set, self.validation_set, self.testing_set)
        delta_t = (datetime.datetime.now() - ping)
        delta_t = delta_t.seconds + delta_t.microseconds / 1000000.
        print("I pulled " + str(num_data) + " images. It took me " + str(delta_t) + " seconds.")
        print("That's about " + str(num_data / float(delta_t)) + " images per second.")
        print("Data was collected \n")

    def sample_handling(self, sample_location, im_amount):
        # this function reads image by image, adds the label and optionally additional information
        # everything is stored in a dictionary
        # each error class however shall be represented in training, validation and test set, so that the images
        # are added to the dicitonary class by class
        # images are skipped if the image limit is reached
        # the functions always read the same data, for same data definition = ensures comparison between different nets
        # for an example refer to the end of the file

        # following sets are the final dictionaries of one component
        train_set = {"features": [], "label": [], "id": [], "add_info": []}
        vali_set = copy.deepcopy(train_set)
        test_set = copy.deepcopy(train_set)
        augmented_images = copy.deepcopy(train_set)

        # Over the development time the image and information storage was changed several times
        # trying to prevent that old versions are used
        # the last supported (of April2018) is one folder containing components, which themselves contain
        # image folders for each class, and the parent folder contains
        # a new_data.dat file containing all the information
        # for each class (like label, id, imagename, angle, bodywidht, etc)
        if os.path.isfile(sample_location+'/new_data.dat'):
            data_name = sample_location+'/new_data.dat'
        elif self.add_info == []:
            print("Warning: You are using the old data.dat format. New one is available.")
            data_name = sample_location+'/data.dat'
        else:
            raise AttributeError("You are using an old format of data.dat. This one is not supported anymore.")

        # are the indices for name, id, angle and label - used from previous implementation
        # alternative represntation can be desirable (might be confusing)
        indices = [0, 1, 2, 3]

        # the classes ratio is recalculated according to the amount of images
        # so instead of, for instance, [1, 2, 1, 1, 0, 1], it will look like
        # [2500, 5000, 2500, 2500, 0, 2500] (so containing the actual amount of images needed)
        cl_ratio = self.adjust_ratio(self.class_ratio, im_amount)
        # then the classes and their corresponding amount of required images is reorder
        # from minimum to maximum
        # that ensure that if one class does not fulfill the amount of images needed
        # the following classes will be recalculated accordingly (so that data_amount is fulfilled)
        classes_tmp, cl_ratio = self.reorder_classes(sample_location, cl_ratio)
        current_class_index = 0
        current_class_amount = 0
        da_ind_tr = 0
        # time to read the images and their information
        with open(data_name, 'r') as data_file:
            # done line by line
            for line in data_file:
                line = line.strip()
                line = line.split(',')

                # rea the first important information
                image_name = str(line[indices[0]])
                image_id = str(line[indices[1]])
                angle = float(line[indices[2]])
                string_tag = str(line[indices[3]])
                if string_tag == 'OK':
                    tag = [0, 1]
                else:
                    tag = [1, 0]

                # if the error tag of the current line does not correspond to the previous data shall be sorted
                # into train, validation and test, added, and the class ratio readjusted
                # Alternative explanation: if the class changes data is read and ratio recalculated
                if string_tag != classes_tmp[current_class_index]:

                    # Data augmentation
                    no_aug_img = 0
                    tr_img_collected = len(train_set["id"]) - da_ind_tr
                    while cl_ratio[current_class_index] > current_class_amount + no_aug_img\
                            and tr_img_collected > no_aug_img and self.augment_data:
                        # print(len(train_set["id"]), len(vali_set["id"]), len(test_set["id"]))
                        features = train_set["features"][da_ind_tr].rotate(180)
                        augmented_images["features"].append(features)
                        augmented_images["label"].append(train_set["label"][da_ind_tr])
                        augmented_images["id"].append(train_set["id"][da_ind_tr])
                        augmented_images["add_info"].append(train_set["add_info"][da_ind_tr])
                        da_ind_tr += 1
                        no_aug_img += 1

                    da_ind_tr = len(train_set["id"])
                    # write the amount of data extracted into classes ratio list
                    cl_ratio[current_class_index] = current_class_amount + no_aug_img
                    current_class_amount = 0
                    # and rewrite the ones which were not read; so that occurs if two classes are 0
                    # then just the last one would be written to 0
                    # this if condition rewrites all the ones which were no read to 0
                    if current_class_index + 1 < classes_tmp.index(string_tag):
                        current_class_index += 1
                        for i in range(current_class_index, classes_tmp.index(string_tag)):
                            cl_ratio[i] = 0

                    # find out which index we are at in the classes list
                    current_class_index = classes_tmp.index(string_tag)
                    # time to recalculate the ratios in each class
                    # Recall: data_amount is highest priority - therefore if not enough images were read for one class
                    # the amount of images for following classes will be readjusted
                    # so that total number will be maximized
                    if self.priority == "data_amount":
                        cl_ratio[current_class_index:] = self.adjust_ratio(
                            cl_ratio[current_class_index:],
                            im_amount -
                            (self.calc_data_amount(train_set, vali_set, test_set)
                             + len(augmented_images["id"])))

                # data is read if the ratio class allows reading from this class
                # Recall: class ratio defines how many images of one class shall be read
                # the data set contains way more images then available
                if current_class_amount < cl_ratio[current_class_index]:
                    # extracting additional information if required
                    add_info = self.add_additonal_information(line)

                    # location of the images
                    sample_location_loop = sample_location+string_tag+"/"

                    features = self.data_initialization(Image.open(sample_location_loop + image_name), angle)

                    # add information to sets
                    # the different if-conditions try to add components evenly to the training, validation and test set
                    if len(train_set["id"]) <= len(vali_set["id"]) * \
                            (1 - self.validation_size - self.test_size) / \
                            self.validation_size:
                        train_set["features"].append(features)
                        train_set["label"].append(tag)
                        train_set["id"].append(image_id)
                        train_set["add_info"].append(add_info)
                    elif len(vali_set["id"]) < len(test_set["id"]) * self.validation_size / self.test_size:
                        vali_set["features"].append(features)
                        vali_set["label"].append(tag)
                        vali_set["id"].append(image_id)
                        vali_set["add_info"].append(add_info)
                    else:
                        test_set["features"].append(features)
                        test_set["label"].append(tag)
                        test_set["id"].append(image_id)
                        test_set["add_info"].append(add_info)
                    current_class_amount += 1
                else:
                    # if data maximum is reached already the following images are ignored
                    pass

        # Data augmentation
        no_aug_img = 0
        tr_img_collected = len(train_set["id"]) - da_ind_tr
        while cl_ratio[current_class_index] > current_class_amount + no_aug_img \
                and tr_img_collected > no_aug_img and self.augment_data\
                and im_amount > self.calc_data_amount(train_set, vali_set, test_set):
            # print(len(train_set["id"]), len(vali_set["id"]), len(test_set["id"]))
            features = train_set["features"][da_ind_tr].rotate(180)
            augmented_images["features"].append(features)
            augmented_images["label"].append(train_set["label"][da_ind_tr])
            augmented_images["id"].append(train_set["id"][da_ind_tr])
            augmented_images["add_info"].append(train_set["add_info"][da_ind_tr])
            da_ind_tr += 1
            no_aug_img += 1
        # write the amount of data extracted into classes ratio list
        cl_ratio[current_class_index] = current_class_amount + no_aug_img

        train_set["features"] += augmented_images["features"]
        train_set["label"] += augmented_images["label"]
        train_set["id"] += augmented_images["id"]
        train_set["add_info"] += augmented_images["add_info"]

        for i in range(len(train_set["features"])):
            train_set["features"][i] = np.array(train_set["features"][i])/255.0
        for i in range(len(vali_set["features"])):
            vali_set["features"][i] = np.array(vali_set["features"][i])/255.0
        for i in range(len(test_set["features"])):
            test_set["features"][i] = np.array(test_set["features"][i])/255.0

        # all data is read and can be returned
        print('Finished reading: '+str(sample_location))
        return train_set, vali_set, test_set

    @staticmethod
    def calc_data_amount(tr, va, te):
        # how much data does the combined set of training validation and test contain?
        return len(tr["id"]) + len(va["id"]) + len(te["id"])

    def reorder_classes(self, location, cl_r):
        # reorders the class depended on the amount of image per class starting with the minimum
        # is needed so that adjust_ratio works properly
        features = []
        amount = []
        for tag in self.classes:
            lst = os.listdir(location + '/' + tag)  # dir is your directory path
            number_files = len(lst)
            amount.append(number_files)

        for a, b, c in zip(self.classes, cl_r, amount):
            features.append((a, b, c))

        features = sorted(features, key=lambda amount: amount[2])

        cla, cl_r = [], []
        for item in features:
            cla.append(item[0])
            cl_r.append(item[1])

        return cla, cl_r

    def data_initialization(self, im, an):
        # resize the pic on defined dimensions
        im = im.resize((self.width, self.height), Image.BILINEAR)
        # Cuts edges to achieve higher information density of the picture itself; dependent on cropping value
        w, h = im.size

        if self.cropping != 0.0:
            im = im.crop((int(w * self.cropping), int(h * self.cropping),  # cuts cropping percent on all edges of pic
                          int(w * (1 - self.cropping)), int(h * (1 - self.cropping))))

        # auto contrast sets lowest value to 0 and highest to 256
        im = ImageOps.autocontrast(im)
        # rotates the image around given angle
        # intuation: all ok images will have a certain orientation
        # therefore looking the same, every other will look different
        features = im.rotate(-an)

        return features

    def add_additonal_information(self, line):
        add_info = []
        for item in self.add_info_indices:
            if item == self.info_list.index("toolname"):  # tool contains a vector
                val = self.tools[line[item]]
            elif item != self.info_list.index("compdimsamples"):  # initialised value; roughly between 0 and 1
                val = [float(float(line[item]) - self.max_min_list[item][1]) /
                       float(self.max_min_list[item][0] - self.max_min_list[item][1])]
            else:  # compdimsamples however shall be either 0 or 1 (and nothing in between; hence boolean type)
                if float(line[item]) > 0:
                    val = [1.0]
                else:
                    val = [0.0]
            # add information to add(itional) info(rmation)
            add_info += val
        return add_info

    @staticmethod
    def adjust_ratio(ratio, im_amount):
        # adjust ratio is based on the assumption that the data amount is highest priority
        # that means that if there are not as many images available as a class OR a component requires
        # that the following classes or components, respectively, are readjusted so that the total amount of
        # images are kept constant - for an exmaple refer to the end of the file
        ratio_sum = sum(ratio)
        for i in range(len(ratio)):
            ratio[i] = \
                int(ratio[i] * im_amount / ratio_sum)
        # using integers the resulting amount could miss some images
        # therefore an image is added; starting by the next following class/component
        for j in range(im_amount - sum(ratio)):
            ratio[j] += 1
        return ratio

    @staticmethod
    def calc_progress(count, total, perc):
        if count % (1 + int(total / (100/perc))) == 0:
            print(str(round(float(count) / total * 100, 2)) + " % done.")

if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except IndexError:
        print("Format needs to be: pic_loader.py <filename.pickle>")
        exit()

# an example should provid clearity: 
# Image we require 1000 images of component 01005 and 1000 images of 0201; we have 10 classes and each class
# should contain equally amount of images; that means 100 images per class
# However, classes like 'SPINNING' or 'DAMAGED' usually do not contain enough images; Image now that
# we could extract 10 Images of 'SPINNING' component 01005 - then to achieve the 1000 images required 
# for component 01005; all other classes need to be recalculated, hence each following class will contain 110 images
# (110 * 9 + 10 = 1000); Image now, that all other classes just have 100 images, so we end up extracting 10 images for
# SPINNING and 100 for all others equaling 990 images - the remaining 10 images missing will then be extracted from
# component 0201, therefore the ratio of components will be [990, 1010]; the ratio for classes in component 01005 is
# [10, 100, 100, 100, 100, 100, 100, 100, 100, 100] 