# NAME
#  runs the classifier training
#
# HELP
#   help for additional commands can be found via 'python main.py -h'
#
# DESCRIPTION
#    Beforehand specifications for different training runs need to be defined
#    For more information see description of specification options further down
#    All specified networks are then trained
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
import datetime
import os
import time
import argparse

# Third-party libraries

# Own package
from data_loader import DataLoader
from parameters import NetworkSpecification
from neural_network import NeuralNetwork
from add_ons import save_specifications


# receive arguments
parser = argparse.ArgumentParser(description='This file defines a simple neural network to classify '
                                             'component pictures. The output are either ok or not ok.')
parser.add_argument('-s', '--save', action='store_true',
                    help='Saves network and accuracy over time (see TensorBoard). '
                         'Automatically sets the name of the to-be-saved file.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Evaluates a test version. Low epoch and almost no input data.')
parser.add_argument('-f', '--finish', action='store_true',
                    help='Will shut down the computer after the program has run.')
args = parser.parse_args()

# parameters: "key": default value,                     explanation
#
#             "validation_size": 0.05,                  Percentage of validation set size of the full data set
#             "test_size": 0.05,                        Percentage of test       set size of the full data set
#             "batch_size": 128,                        How many images in one batch (integer)
#             "total_epochs": 200,                      Number of epochs run
#             "dimension": (32, 32),                    Amount of pixels; (width, height)
#             "cropping": 5,                            How many pixels to crop on each side
#             "optimizer_name": "Adam",                 Which optimizer for learning; "SGD" or "Adam"
#             "learning_rate": 0.001,                   Learning rate for the optimizer
#             "activation_function": "relu",            Which activation function for layers; "relu","sigmoid" or "tanh"
#             "fc_neurons": [150, 75],                  Amount of neurons per fully connected layer
#             "conv_neurons": [],                       Definition of convolutional layer; exm.: [(4, 8), (4, 16)]
#                                                                                   = [(filter_size, amount of filters)]
#             "conv_pad": "NEVER",                      How often apply 'VALID' padding - default 'SAME';
#                                                       "ALWAYS", "FIRST", "FIRST2" or "NEVER"
#             "regularization": False,                  Shall L2 regularization be applied (True) or not (False)
#             "beta": 0.01,                             Regularization parameter
#             "early_stopping": True,                   Shall early stopping be applied or not
#                                                       After 20 epochs without improvement of cost func on validation
#             "early_stopping_costumized": False,       Beta version; not tested propperly yet
#             "learning_rate_decay": False,             Decreases learning rate every 10 epochs if no new train loss min
#             "pooling": [2, 2],                        Pooling parameters [step size/stride, filter size]
#             "tensorboard_dir": "/test/",              Which sub folder in tensorboard is used
#             "case_dir": "/test/",                     Describes what is the overall theme of the run; exm.: batch_size
#             "data_amount": 200000                     How much data shall be collected
#             "data_dir": '~/data/',                    Where are the image component folders saved
#             "components": ['01005/', '/Large/',       Which components to read
#                            '0201/', '0805/', '1206/',
#                            '0603/', '0402/'],
#             "components_ratio": [1, 1, 1, 1, 1, 2, 2], Data ratio between the different component classes
#             "classes": ["OK", "NOK"],            Defines which error classes to use
#             "classes_ratio": [1, 1],                    Data ratio between the different error classes
#             "data_priority": "data_amount",           Data collection priority:
#                                                       "data_amount" = gap of images in minor classes are filled up
#                                                                       with images from major classes
#                                                       "components_ratio"       = keep the defined component ratios
#                                                       "classes_ratio"       = keep the defined classes ratios
#             "augment_data"                            Augments the data set by rotation the images around 180 degrees
#             "gpu": 1,                                 Decides which GPU to use
#                                                       if specified wrongly, tensorflow selects automatically
#             "add_info": [],                           Add additional component, package information to layer;
#                                                       ["all"] includes all available information
#             "add_info_layer": 1,                      Defines layer where to concatenates additional information;
#                                                       Here, 0 is input layer, 1 is first hidden layer, etc
#             "add_string": "",                         Add string at the back of the network's name
#             "collect_data": False,                    Automatically set
#             "num_images": [99120, 86598]              The amount of ok and not ok pictures


# Options applied to all networks
# example:
# general_options = {"dimension": (34,34),
#                    "cropping": 0}
general_options = {}

additional_options = list()
# defines network specific options
# example:
# additional_options.append({"regularization": True,
#                            "beta": 0.02,
#                            "add_info": "_L2regu-beta002"})
# additional_options.append({"conv_neurons": [(4, 8), (8, 6)]})
additional_options.append({})

# For development purposes a test run can be executed
# instead of reading all data and runing for lots of epochs this one is executed quicker
if args.test:
    general_options["total_epochs"] = 20
    general_options["data_amount"] = 2000
    general_options["early_stopping"] = False
    general_options["learning_rate_decay"] = True
    if args.save:
        general_options["tensorboard_dir"] = "/test/"

# Sets up the information of the network in file 'parameter.py'
networks = NetworkSpecification(general_options, additional_options)

# Defines the GPU on which network is trained
# no run specefic option
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(networks.specification[0]["gpu"])

# For time performance evaluation
ping1 = datetime.datetime.now()
# Run through all specified networks
for specs in networks.specification:
    if specs["collect_data"]:   # is called if data specification changes
                                # ensures that for every run the data is read again
        # function found at 'pic_loader.py'
        loader = DataLoader(specs)
        loader.load_data()
        # training_set, validation_set, test_set = \
        #     load_data(specs)
    else:
        print("No change in data format. Using the same data ...")
    # saving information about the network in a subfolder; will be loaded afterwards
    if args.save:
        save_specifications(specs["file_dir"], specs['name'], specs)

    # setting up network in file 'neural_network.py'
    net = NeuralNetwork(specs, args.save)
    # time to Train Yuhu !
    net.train_neural_network(loader.training_set, loader.validation_set, loader.testing_set)


print("The whole thing took me " + str((datetime.datetime.now()-ping1).seconds) + " seconds.")

# can shut down the computer after the run is completed
if args.finish:
    print('\n \n \n \n The computer will be switched off in 30 minutes.')
    time.sleep(30*60)
    os.system("shutdown now -h")
