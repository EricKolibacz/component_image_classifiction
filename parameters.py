# NAME
#  Defines the parameters(specifications) for each run
#
# HELP
#   usually not called directly (no main call)
#
# DESCRIPTION
#    Sets up information about the networks
#    General options are valid for all networks; additional are network specific
#    Additional information overwrite general options
#    Some of the parameters are rewritten (e.g. components_ratio, classes_ratio) for later porpuses
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
import os
import copy
# Third-party libraries

# Own package
from add_ons import create_name


class NetworkSpecification(object):
    def __init__(self, general, additional, print_screen=True):
        # default parameters which will be applied of nothing differently specified
        # Explanation of ever single on of them in 'run.py'
        self.default_init = {
            "validation_size": 0.05,
            "test_size": 0.05,
            "batch_size": 128,
            "total_epochs": 200,
            "dimension": (32, 32),
            "cropping": 5,
            "optimizer_name": "Adam",
            "learning_rate": 0.001,
            "activation_function": "relu",
            "fc_neurons": [150, 75],
            "conv_neurons": [],
            "conv_pad": "NEVER",  # Refers to how often apply 'VALID' padding - default 'SAME'
            "regularization": False,
            "beta": 0.01,         # Regularization parameter
            "early_stopping": True,
            "early_stopping_costumized": False,
            "learning_rate_decay": False,
            "pooling": [2, 2],
            "tensorboard_dir": "/test/",
            "case_dir": "/test/",
            "data_amount": 200000,
            "data_dir": '~/data/',
            "components": ['/01005/', '/Large/', '/0805/', '/0201/',
                           '/1206/', '/0603/', '/0402/'],   # Sorted by size
            "components_ratio": [1, 1, 1, 1, 1, 2, 2],
            "classes": ["NOK", "OK"],
            "classes_ratio": [1, 1],
            "data_priority": "data_amount",
            "augment_data": False,
            "gpu": 1,
            "add_info": [],
            "add_info_layer": 1,
            "add_string": "",
            "collect_data": False,
            "num_images": [99120, 86598]
        }

        # Note 'copy.deepcopy'; therefore the dictionary is properly copied; python specific 
        self.default = copy.deepcopy(self.default_init)

        # Ensuring that the specifications exists; otherwise programs ends with error
        for key in general:
            if key not in self.default_init.keys():
                raise AttributeError("Specified key " + str(key) +
                                     " of general option list is not a valid option.")
        for option in additional:
            for key in option:
                if key not in self.default_init.keys():
                    raise AttributeError("Specified key " + str(key) +
                                         " of additional option list is not a valid option.")

        # Over write default options with general options
        for key in general:
            self.default[key] = general[key]

        # each network is saved in a dictionary; together they are collected in a list
        self.specification = list()

        # Over write run specific options
        for option in additional:
            run_spec = copy.deepcopy(self.default)
            for key in option:
                # Maybe redundant ???
                if key in run_spec:
                    run_spec[key] = option[key]
                else:
                    raise AttributeError("The entered key '" + str(key) + "' is not in the specification list.")

            # Ensures that data will be read JUST if some data definition changes
            # That is also the case if a default data definition follows a modified one
            if self.collect_data_flag(run_spec):
                run_spec["collect_data"] = True

            # Finds the parent directory of 'master_dl'; usually tensorboard is located in the same
            string = os.path.realpath(__file__)
            run_spec["file_dir"] = string[:string.rfind("master_dl")] + '/tensorboard/' + \
                                   str(run_spec["tensorboard_dir"]) + str(run_spec["case_dir"])

            # Redefines components_ratio to exact amount of needed images per component
            run_spec["components_ratio"] = self.data_details(run_spec["components_ratio"], run_spec["data_amount"])
            # Redefines the classes ratio if OK and NOK specified
            run_spec["classes"], run_spec["classes_ratio"] = self.error_details(run_spec["classes"],
                                                                                run_spec["classes_ratio"])
            # Create network name; defined in 'addons.py'
            # The name includes how many OK and NOK images are included; exact definition seems complicated
            run_spec["name"] = create_name(run_spec["dimension"][0], run_spec["dimension"][1], run_spec['cropping'],
                                           run_spec["fc_neurons"],
                                           run_spec["conv_neurons"], data_ok=
                                           int(run_spec["data_amount"]*run_spec["classes_ratio"][-1]
                                               / sum(run_spec["classes_ratio"])),
                                           data_error=
                                           int(run_spec["data_amount"]*sum(run_spec["classes_ratio"][:-1])
                                               / sum(run_spec["classes_ratio"])))\
                                + run_spec["add_string"]
            # Prints the name of the network
            if print_screen:
                print("Network name: " + str(run_spec["name"]))

            self.specification.append(run_spec)

    def collect_data_flag(self, current_spec):
        # if data specific specification changes with respect to prior run flag is set to true
        if self.specification:
            status = False
            # specifications which have influence on the read data; COMPLETE ??? 
            data_related_keys = ["cropping", "dimension", "components", "validation_size", "test_size",
                                 "data_amount", "components_ratio", "data_priority", "augment_data"]
            past_spec = self.specification[-1].copy()
            for item in data_related_keys:
                # compare current specification with prior
                if current_spec[item] != past_spec[item]:
                    status = True
        else:   # enters for first network
            status = True
        return status

    def data_details(self, components_ratio, data_amount): # redefines the ratio between images per component
        ratio_sum = sum(components_ratio)
        for i in range(len(components_ratio)):
            components_ratio[i] = \
                int(components_ratio[i]*data_amount/ratio_sum)
        for j in range(1, data_amount - sum(components_ratio)+1):
            components_ratio[-j] += 1

        return components_ratio

    def error_details(self, classes, classes_ratio): # redefines the ratio between images per class
        if "NOK" in classes: # If classification OK and NOK is defined 
            ok_ratio = classes_ratio[classes.index("OK")]
            nok_ratio = classes_ratio[classes.index("NOK")]

            classes = ["SPINNING", "DAMAGED", "WRONG PICK ANGLE", "TOMBSTONED", "CORNED PICK",
                       "BILLBOARDED", "UPSIDE DOWN", "NOT PICKED", "STOP PRODUCTION", "OK"]
            classes_ratio = [nok_ratio for _ in range(len(classes))]

            classes_ratio[-1] = ok_ratio * len(classes[:-1])

        return classes, classes_ratio
