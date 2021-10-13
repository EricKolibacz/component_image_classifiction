"""Set up neural network, its layers, activation functions and padding.

Author: Eric Kolibacz"""

# Libraries
# Standard libraries
# Third-party libraries
import tensorflow as tf
#Own package


def neural_network_model(data, add_x, add_x_layer, classes, fc_neurons, conv_neurons, conv_pad, activation_function, pooling, width, height):
    # This function sets up the architecture of the neural network.
    layer_act = [data]  # computes the neurons output, structure [x, a1, a2, ..., a_L]
    weights = []

    # set up convolutional layers (including pooling)
    if conv_neurons:
        conv_def = [(5, 1)] + conv_neurons

        for i in range(1, len(conv_def)):
            conv, weight = conv_layer(layer_act[-1], conv_def[i - 1][1], conv_def[i][0], conv_def[i][1],
                                      pooling, conv_pad=define_padding(layer_act, conv_pad),
                                      activation=activation_function, name="conv" + str(i))
            layer_act.append(conv)
            weights.append(weight)

        # last multidimensional conv layer will be flatten to be able to connect to fc layer
        flat, num_neurons = flatten_layer(layer_act[-1])
        layer_act.append(flat)
        hidden_layers = [num_neurons] + fc_neurons
    else:
        hidden_layers = [width * height] + fc_neurons

    # set up fully connected layers
    for i in range(len(hidden_layers) - 1):
        if i == add_x_layer:
            layer_in = tf.concat([layer_act[-1], add_x], 1)
        else:
            layer_in = layer_act[-1]
            
        fc, weight = fc_layer(layer_in, layer_in.shape[1], hidden_layers[i + 1],
                              activation=activation_function, name="fc" + str(i))
        layer_act.append(fc)
        weights.append(weight)

    output, weight = fc_layer(layer_act[-1], hidden_layers[-1], classes, activation="", name="output_layer")
    weights.append(weight)

    return output, weights


def fc_layer(input, channels_in, channels_out, activation="relu", name="fc"):
    ''' Sets up a fully connected layer. input is the input data to the layer. Channels_in
    sets the amount of neurons in the incoming layer, channels_out is the amount of the neurons
    to the connected layer. Activation flag will apply the activation function to z. '''
    with tf.name_scope(name):  # names the scope for TensorBoard
        # with respect to andrew ng: Different initialization for different activation
        initializer = set_initializer(activation)

        weights = tf.get_variable(name+"W", shape=(channels_in, channels_out), initializer=initializer)
        biases = tf.Variable(tf.random_normal([channels_out]), name="B")

        act = tf.add(tf.matmul(input, weights), biases)
        if activation == 'relu':
            act = tf.nn.relu(act)
        elif activation == 'tanh':
            act = tf.nn.tanh(act)
        elif activation == 'sigmoid':
            act = tf.nn.sigmoid(act)
        if 0:
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", act)

        return act, weights


def conv_layer(input, num_input_channels, filter_size, num_filters, pooling,
               activation="relu", name="conv", conv_pad="SAME"):
    ''' Sets up a convolutional layer. input is the input data to the layer. Channels_in
    sets the amount of neurons in the incoming layer, channels_out is the amount of the neurons
    to the connected layer. Activation flag will apply the activation function to z. '''
    with tf.name_scope(name):  # names the scope for TensorBoard
        # with respect to andrew ng: Different initialization for different activation
        initializer = set_initializer(activation)

        shape = (filter_size, filter_size, num_input_channels, num_filters)

        weights = tf.get_variable(name+"W", shape=shape, initializer=initializer)
        biases = tf.Variable(tf.random_normal([num_filters]), name="B")
        # strides as mentioned here https://www.youtube.com/watch?v=HMcx-zY8JSg
        # image number, x-axis, y-axis, image layer - 1 step each (first and last is set)
        # same padding="SAME" will result in same dimension for input and output image
        conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=conv_pad)
        conv = tf.add(conv, biases)

        if pooling:
            # just works for max pooling and only increasing activation function
            # alternatively this must be executed afterwards
            # Here, however, it reduces the calculation time
            conv = tf.nn.max_pool(value=conv,
                                  ksize=[1, pooling[0], pooling[0], 1],
                                  strides=[1, pooling[1], pooling[1], 1],
                                  padding='SAME')

        if activation == 'relu':
            act = tf.nn.relu(conv)
        elif activation == 'tanh':
            act = tf.nn.tanh(conv)
        elif activation == 'sigmoid':
            act = tf.nn.sigmoid(conv)
        else:
            act = conv
        if 0:
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", act)

        return act, weights


def flatten_layer(layer):
    # flattens the output of the last conv layer to connect it to fully connect layers
    # Credit https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # layer_shape == [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def set_initializer(activa):
    # different initializer depending on the activation function are defined
    # source online tutorial andrew ng
    if activa == 'relu':
        factor = 2.0
        mode = 'FAN_IN'
    elif activa == 'tanh':
        factor = 1.0
        mode = 'FAN_IN'
    elif activa == 'sigmoid':
        factor = 1.0
        mode = 'FAN_IN'
    else:  # output layer
        factor = 1.0
        mode = 'FAN_IN'

    return tf.contrib.layers.variance_scaling_initializer(
        factor=factor,
        mode=mode)


def define_padding(layer, status):
    # defines the padding for a conv or pooling
    # default 'SAME' shapes zeros equally around the image (fo runeven right and bottom 1 more)
    pad = 'SAME'
    if status == 'ALWAYS':
        pad = 'VALID'
    elif status == 'FIRST':
        if len(layer) == 1:
            pad = 'VALID'
    elif status == 'FIRST2':
        if len(layer) <= 2:
            pad = 'VALID'

    return pad
