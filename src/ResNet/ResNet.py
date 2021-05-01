import tensorflow as tf
tf.config.run_functions_eagerly(True)

class ResnetBase:

    '''
    INITIALISER
    Creates an object ResNet which can be trained as a keras model (with tensorflow backend)

    ARGS:
    input_shape is the shape ( as a tuple/list ) of the feature vector X
    activation is a keras activation function, relu by default
    
    '''
    def __init__(self, input_shape,activation=tf.keras.layers.Activation('relu')):
        self.activation = activation
        self.input_shape = input_shape
        # self.classes = classes
        self.X_input = tf.keras.layers.Input(input_shape)
        self.X = self.X_input


    '''
    ADD_IDENTITY_BLOCK (RESIDUAL_BLOCK) where input and output shapes are SAME
    Creates and Adds identity block to the network. 
    The first and the last layers in this block are connected
    by an X_shortcut.

    ARGS:
    layers: list of the intermediate layers
    activation: activation function. default is the gloabl default for ResnetBase object
    '''
    def add_identity_block(self, layers, activation=None):

        if activation is None:
            activation = self.activation

        X_shortcut = self.X
        for layer in layers[:-1]:
            self.X = layer(self.X)
            self.X = activation(self.X)

        self.X = layers[-1](self.X)

        self.X = tf.keras.layers.Add()([self.X, X_shortcut])
        self.X = activation(self.X)


    '''
    ADD_CONV_BN_BLOCK where input and output shapes MAY or MAY NOT be SAME
    Creates a block of conv2D layers followed by batch normalisation and activation function

    
    '''
    def add_conv_bn_block(self, kernel_sizes, filters, strides=(1,1), shortcut=True, activation=None):

        if activation is None:
            activation = self.activation

        X_shortcut = self.X

        self.X = tf.keras.layers.Conv2D(kernel_size = kernel_sizes[0], filters=filters[0], strides=strides, padding="same")(self.X)
        self.X = tf.keras.layers.BatchNormalization(axis=3)(self.X)
        self.X = activation(self.X)

        if len(filters) > 1:
            for i in range(1,len(filters)-1):
                    self.X = tf.keras.layers.Conv2D(kernel_size=kernel_sizes[i], filters=filters[i], padding="same")(self.X)
                    self.X = tf.keras.layers.BatchNormalization(axis=3)(self.X)
                    self.X = activation(self.X)

        X_shortcut = tf.keras.layers.Conv2D(kernel_size= kernel_sizes[0], filters=filters[-1], strides=strides, padding="same")(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)

        self.X = tf.keras.layers.Conv2D(kernel_size=kernel_sizes[-1], filters=filters[-1], padding="same")(self.X)
        self.X = tf.keras.layers.BatchNormalization(axis=3)(self.X)
        self.X = tf.keras.layers.Add()([self.X, X_shortcut])
        self.X = activation(self.X)

    '''
    ADD_LAYERS
    Adds layers to the model.

    ARGS:
    layers: list of layers to be added OR a single layer. 
    activation: activation function
    '''
    def add(self, layers, activation=None):
        if type(layers) is not list and type(layers) == tf.keras.layer:
            layers = [layers]
        for layer in layers:
            self.X = layer(self.X)

    '''
    returns a keras model which needs to be compiled, fit and tested similar to any inbuilt keras model.
    '''
    def build_model(self):
        self.model = tf.keras.models.Model(inputs=self.X_input,  outputs=self.X, name='ResNet')
        return self.model
