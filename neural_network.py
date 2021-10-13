# NAME
#  neural network
#
# HELP
#   no help information provided since called indirectly
#
# DESCRIPTION
#    sets up the network and runs the trainingd
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
import math

# Third-party libraries
import numpy as np
import tensorflow as tf

# Own package
from add_ons import time_estimation, shuffle_data
from define_layers import neural_network_model

# Defines a neural network - the only function so far is 'training'
class NeuralNetwork(object):
    def __init__(self, specs, save):

        # All prior defined specifications are now saved to the network class
        self.name = specs["name"]

        self.batch_size = specs["batch_size"]
        self.total_epochs = specs["total_epochs"]

        self.optimizer_name = specs["optimizer_name"]
        self.learning_rate = specs["learning_rate"]
        self.activation_function = specs["activation_function"]

        self.fc_neurons = list(specs["fc_neurons"])
        self.conv_neurons = list(specs["conv_neurons"])
        self.conv_pad = specs['conv_pad']
        self.pooling = specs["pooling"]

        self.regularization = specs["regularization"]
        self.beta = specs["beta"]

        self.components = specs["components"]
        self.tensorboard_dir = specs["tensorboard_dir"]
        self.cropping = specs["cropping"]
        self.width = specs["dimension"][0]-2*self.cropping
        self.height = specs["dimension"][0]-2*self.cropping

        self.gpu = specs["gpu"]

        self.add_info = specs["add_info"]
        self.add_info_layer = specs["add_info_layer"]

        # Includes information of saving, early stopping, time_estimation or not
        self.flag = {"early_stopping": specs["early_stopping"],
                     "early_stopping_costumized": specs['early_stopping_costumized'],
                     "learning_rate_decay": specs['learning_rate_decay'],
                     "time_estimation": True,
                     "save": False,
                     "save_net": False}

        if save:
            self.file_dir = specs["file_dir"]

            self.flag["save"] = True

    # Training the network - only information needed is the data
    def train_neural_network(self, tr, va, te):
        if self.flag["save"]:
            print("Classifier "+str(self.name)+" will be tuned.\n")

        train_x = tr["features"]
        train_y = tr["label"]
        train_name = tr["id"]
        train_ai = tr["add_info"]

        tf.reset_default_graph()
        # all calculations are saved and performed on one GPU - is garantied in run.py already
        with tf.device('/device:GPU:'+str(self.gpu)):

            # input
            x = tf.placeholder('float', [None, self.width, self.height], name="x")
            # if CNN are used (so the list is non-empty) the input is an image - otherwise a vector
            if self.conv_neurons:
                x_net = tf.reshape(x, [-1, self.width, self.height, 1])
            else:
                x_net = tf.reshape(x, [-1, self.width * self.height])
            # output 
            y = tf.placeholder('float', [None, len(train_y[0])], name="labels")
            # additional information as input
            a = tf.placeholder('float', [None, len(train_ai[0])], name="a")

            # setting up the network; conv layers, fully conected layers, which activiation, etc
            prediction, weights = \
                neural_network_model(x_net, a, self.add_info_layer,
                                     len(train_y[0]), self.fc_neurons, self.conv_neurons, self.conv_pad,
                                     self.activation_function, self.pooling, self.width, self.height)

            # cost function with cross entropy (mean is an alternative)
            with tf.name_scope("cost"):  # names the scope for TensorBoard
                reg = 0
                if self.regularization:  # if specified the L2-regularization on weights will be done
                    for weight in weights:
                        reg += tf.nn.l2_loss(weight)
                # cost function of the network
                cost = tf.add(
                    tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)),
                    tf.multiply(self.beta, reg))
                # write it to Tensorboard to review it later
                tf.summary.scalar("cost", cost)

            # either Adam Optimizer od SGD; as defined in specifications
            # learning rate can be reduced if defined - more further down
            with tf.name_scope("train"):  # names the scope for TensorBoard
                if self.optimizer_name == "Adam":
                    optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

            # accuracy is here defined as the rate of how many samples are classified correctly
            # this was defined as the evaluation method for binary classifier 
            # See Thesis document for more explanation
            with tf.name_scope("accuracy"):  # names the scope for TensorBoard
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                tf.summary.scalar("accuracy", accuracy)

            merged_summary = tf.summary.merge_all()

        # Some more information can be printed and defined ofr Tensorflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Just use the Memory of the GPU as needed (False = everything)
        config.allow_soft_placement = True      # If some of the calculations require to be placed on CPU
                                                # then this is done automatically
        config.log_device_placement = False     # prints where each calculation (as cost, accurac, etc) is placed

        # start the actual training
        with tf.Session(config=config) as sess:
            # initialization is needed; otherwise errors
            sess.run(tf.global_variables_initializer())
            # define the tensorboard writer for training, validation and test
            # set the folder as subfolder of the network folder
            if self.flag["save"]:
                train_writer = tf.summary.FileWriter(self.file_dir+str(self.name)+'/train')
                train_writer.add_graph(sess.graph)
                val_writer = tf.summary.FileWriter(self.file_dir+str(self.name)+'/validation')
                val_writer.add_graph(sess.graph)
                test_writer = tf.summary.FileWriter(self.file_dir+str(self.name)+'/test')
                test_writer.add_graph(sess.graph)

            # A step is needed to be able to compare different trained networks
            tensorboard_step = 0
            # these varibales help to calculate the training accuracy and cost
            # if all data used it will crash the program, if just one batch it is not accurate enough (in 1/batch_size percente)
            tensor_batch_x = np.empty([0, self.width, self.height])
            tensor_batch_ai = np.empty([0, len(train_ai[0])])
            tensor_batch_y = np.empty([0, len(train_y[0])])
            # Helps early stopping
            min_val_cost_epoch = 0
            min_val_cost = 10**10
            # Helps learning rate adjustment
            min_train_cost_epoch = 0
            min_train_loss = 10**10
            # helps to calculate run time estimation which hits after 10 % of run epochs
            # Note does not work with early stopping
            ping = datetime.datetime.now()

            # Time to run all the training epochs
            for epoch in range(self.total_epochs):
                epoch_loss = 0
                i = 0

                # time estimation is done here
                percentage = 0.1
                if epoch >= percentage * self.total_epochs and self.flag['time_estimation']:  # estimate for run time
                    self.flag['time_estimation'] = time_estimation(ping, percentage=percentage)

                # shuffle the training data 
                train_x, train_y, train_name, train_ai = shuffle_data(train_x, train_y, train_name, train_ai)
                
                # Run over the training set, batch by batch
                # last samples are disregarded (if samples % batch_size != 0)
                while i < len(train_x) - self.batch_size:
                    start = i
                    end = i+self.batch_size

                    # grep a batch of samples
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    batch_ai = np.array(train_ai[start:end])

                    # Run the optimizer
                    _, c, _ = sess.run([optimizer, cost, prediction],
                                       feed_dict={x: batch_x, y: batch_y, a: batch_ai})
                    epoch_loss += c

                    i += self.batch_size

                    # Add information for TensorBoard
                    if self.flag["save"]:
                        if tensorboard_step % int(len(train_x)/10000.) == 0:     
                        # reducing the size of summary file so that just every now and then a summary is written
                            tensor_batch_x = np.concatenate((tensor_batch_x, batch_x), axis=0)
                            tensor_batch_ai = np.concatenate((tensor_batch_ai, batch_ai), axis=0)
                            tensor_batch_y = np.concatenate((tensor_batch_y, batch_y), axis=0)
                        if tensorboard_step % int(len(train_x)/250.) == 0:
                        # reducing the size of summary file so that just every now and then a summary is written
                            s = sess.run(merged_summary,
                                         feed_dict={x: va["features"], y: va["label"], a: va["add_info"]})
                            val_writer.add_summary(s, tensorboard_step)
                            s = sess.run(merged_summary,
                                         feed_dict={x: te["features"], y: te["label"], a: te["add_info"]})
                            test_writer.add_summary(s, tensorboard_step)
                            s = sess.run(merged_summary,
                                         feed_dict={x: tensor_batch_x, y: tensor_batch_y, a: tensor_batch_ai})
                            train_writer.add_summary(s, tensorboard_step)
                            # clear the training batch variables
                            tensor_batch_x = np.empty([0, self.width, self.height])
                            tensor_batch_ai = np.empty([0, len(train_ai[0])])
                            tensor_batch_y = np.empty([0, len(train_y[0])])

                    tensorboard_step += 1

                # early stopping if 20 epochs did not gain any improvements on validation cost
                if self.flag['early_stopping']:
                    # calculate current validation cost
                    current_val_cost = cost.eval({x: va["features"], y: va["label"], a: va["add_info"]})
                    if current_val_cost < min_val_cost: # if new minimum of cost on validation overwrite min
                        min_val_cost_epoch = epoch
                        min_val_cost = current_val_cost
                        if self.flag['save_net']: # this flag hits if an improvement hits after 5 epochs of non-imporvements
                            self.save_network(sess, tensorboard_step)

                    if epoch - min_val_cost_epoch >= 5 and self.flag['save']:   # save the network structure
                                                                                # if an imporvement after 5 non-imporvements
                        self.flag['save_net'] = True

                    # this early stopping is done so that networks with different batch_sizes can be compared
                    # Advisable to set to False
                    if self.flag['early_stopping_costumized']:
                        if epoch - min_val_cost_epoch > 20 * math.sqrt(self.total_epochs/64.):
                            print(20 * math.sqrt(self.total_epochs/64.))
                            print("Early stop at epoch " + str(epoch) + '.\n')
                            break
                    elif epoch - min_val_cost_epoch > 20:
                        print("Early stop at epoch " + str(epoch) + '.\n')
                        break
                # learning rate reduction
                # normalised loss function
                if self.flag["learning_rate_decay"]:
                    # receive a costum batch of samples
                    batch_x = np.array(train_x[:5000])
                    batch_y = np.array(train_y[:5000])
                    batch_ai = np.array(train_ai[:5000])
                    # calculate the cost for the training batch
                    current_train_loss = cost.eval(feed_dict={x: batch_x, y: batch_y, a: batch_ai})
                    if current_train_loss < min_train_loss:
                        # save new minimum on loss - actually redundant
                        min_train_loss = current_train_loss
                    # every quarter of training reduce the learning rate to 25% (coincendece that both is divided by 4)
                    if epoch - min_train_cost_epoch > self.total_epochs / 4:
                        self.learning_rate /= 4.
                        min_train_cost_epoch = epoch
                # Occasionally report accuracy (every 5th epoch)
                if (epoch+1) % 5 == 0:
                    print('Epoch '+str(epoch+1)+' completed out of '+str(self.total_epochs)+' loss: '+str(epoch_loss))
                    print('Accuracy on validation data: ' +
                          str(accuracy.eval({x: va["features"], y: va["label"], a: va["add_info"]})*100)+'\n')

            print('Accuracy on test data: ' +
                  str(accuracy.eval({x: te["features"], y: te["label"], a: te["add_info"]})*100) +
                  " %")
            if self.flag['save'] and not self.flag['save_net']: # hits if the network was not saved prior
                self.save_network(sess, tensorboard_step)

            print("\nYuhu, we are done training this network!\n\n\n\n")

    # defines the saving network routine
    # thereby the weights, biases and so on are saved so that they can be used for cassification lateron
    def save_network(self, sess, t_step):
        saver = tf.train.Saver()
        print('Saving current model...\n')
        # network is saved in a subfolder (called 'network') in the parent network folder
        saver.save(sess,
                   self.file_dir + str(self.name) + '/network/' + str(self.name),
                   global_step=t_step)
