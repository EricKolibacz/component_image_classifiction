"""
time_estimation:

Prints a estimate on how long a training will run at a certain point.


create_name:
Creates network name

example output:
    24x24_lc1-c3-100-50_d100000-44255_crop

neuron_specifications:
    configures layer definition into correct format


Author: Eric Kolibacz"""
# Libraries
# Standard libraries
import datetime
import random
import pickle
import os
import argparse

# Third-party libraries


# Own package


def create_name(width, height, cropping, fc_n, conv_neurons, data_ok=100000, data_error=44255):
    name_dim = str(width)+'x'+str(height)+'c'+str(cropping)

    name_layer = 'l'
    for neurons in conv_neurons:
        name_layer = name_layer+'c'+str(neurons[0])+'f'+str(neurons[1])+'-'

    for neurons in fc_n:
        name_layer = name_layer+str(neurons)+'-'
    name_layer = name_layer[:-1]

    name_data = 'd'+str(data_ok)+'-'+str(data_error)

    name = str(name_dim)+'_'+str(name_layer)+'_'+str(name_data)

    return name


def time_estimation(ping, percentage=0.1):
    experience = 1.366    # experience factor is a value how a process usually takes longer
    pong = datetime.datetime.now()
    expected_time = (pong - ping).seconds/60 * (int(1/percentage)-1) * experience
    print(str(int(percentage*100))+' % of the run is completed. The full run will take maximum another ' +
          str(int(expected_time)) + ' minutes.')
    print('Finishing around ' + str(pong + datetime.timedelta(0, expected_time*60))+'\n')

    return False


def shuffle_data(x, y, n, a):
    if not x:
        raise AttributeError('No list for train_x is given. One is required to be non-empty.')
    features = []
    for a, b, c, d in zip(x, y, n, a):
        features.append([a, b, c, d])

    random.shuffle(features)

    x_out, y_out, name_out, ai_out = [], [], [], []
    for item in features:
        x_out.append(item[0])
        y_out.append(item[1])
        name_out.append(item[2])
        ai_out.append(item[3])

    return x_out, y_out, name_out, ai_out


def split_set(data_set, validating_size, testing_size):

    for i in range(len(data_set["label"])):
        if data_set["label"][i] == 'OK':
            data_set["label"][i] = [0, 1]
        else:
            data_set["label"][i] = [1, 0]

    v_s = int(validating_size*len(data_set["id"]))+1
    t_s = int(testing_size*len(data_set["id"]))+1

    index1 = (v_s+t_s)
    index2 = t_s

    train = {}
    valid = {}
    test = {}

    for key in data_set:
        train[key] = data_set[key][:-index1]
        valid[key] = data_set[key][-index1:-index2]
        test[key] = data_set[key][-index2:]

    return train, valid, test


def save_specifications(location, name, obj):
    fo = location + '/' + name + '/network/'
    if not os.path.exists(fo):
        os.makedirs(fo)
    with open(fo + 'specification.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_specifications(name):
    with open(name + '/specification.pkl', 'rb') as f:
        return pickle.load(f)


def calc_variables(sp):
    x = 0
    if sp["conv_neurons"]:
        for item in sp["conv_neurons"]:
            x += item[0]**2 * item[1] + 1
        input_n = (sp["dimension"][0] - 2*sp["cropping"]) * (sp["dimension"][0] - 2*sp["cropping"]) / \
                  (sp["pooling"][1]**2 ** len(sp["conv_neurons"]))
    else:
        input_n = (sp["dimension"][0] - 2*sp["cropping"]) * (sp["dimension"][0] - 2*sp["cropping"])
    layers = [input_n] + sp["fc_neurons"] + [2]
    for i in range(1, len(layers)):
        x += layers[i-1] * layers[i] + layers[i]
    return int(x)


if __name__ == '__main__':
    from parameters import NetworkSpecification
    # receive arguments
    parser = argparse.ArgumentParser(description='This file contains additional functions for the neural networks.')
    parser.add_argument('-v', '--variables', action='store_true',
                        help='Calculates how many Variables a network contains.')
    parser.add_argument('-f', '--variables-from-file', action="store", dest="file",
                        help='Calculates how many Variables a network contains. '
                             'Reading data from specification.pkl.')

    args = parser.parse_args()
    if args.variables or args.file:
        nets = []
        if args.file:
            networks = load_specifications(args.file.rsplit("/", 1)[0])
            nets.append(networks)
        if args.variables:
            # add your specifications here
            ##########################################################################
            general_options = {"early_stopping": False,
                               "gpu": 0,
                               "case_dir": "/bias_variance/",
                               "total_epochs": 200,
                               "fc_neurons": [2048, 1024, 512]}

            additional_options = list()
            # defines network specific options
            # example:
            additional_options.append({})
            additional_options.append({"regularization": True,
                                       "add_info": "_L2regu"})
            additional_options.append({"conv_neurons": [(4, 8)]})
            additional_options.append({"conv_neurons": [(4, 8), (4, 16)]})
            additional_options.append({"conv_neurons": [(4, 8), (4, 16), (4, 32)]})
            ##########################################################################
            networks = NetworkSpecification(general_options, additional_options, print_screen=False)
            for specs in networks.specification:
                nets.append(specs)
        print("\n")
        text_width = 0
        for specs in nets:
            if text_width < len(specs["name"]):
                text_width = len(specs["name"])
        print(("{:"+str(text_width)+"} {:>15} ").format("Network name", "Variables"))
        for specs in nets:
            print(("{:" + str(text_width) + "} {:>15,} ").format(specs["name"], calc_variables(specs)))
        print("\n")

