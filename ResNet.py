import tensorflow as tf

class Resnet:

    '''
    INITIALISER
    Creates an object ResNet which can be trained as a keras model (with tensorflow backend)
    input_shape is the shape ( as a tuple/list ) of the feature vector X
    actiavtion is a tensorflow activation function, relu by default
    '''
    def __init__(self, input_shape,activation=tf.keras.layers.Activation('relu')):
        self.activation = activation
        self.input_shape = input_shape
        # self.classes = classes
        self.X_input = tf.keras.layers.Input(input_shape)
        self.X = self.X_input


    '''
    ADD_IDENTITY_BLOCK
    Creates and Adds identity block to the network. The first and the last layers in this block are connected
    by an X_shortcut. It takes a list of the intermediate layers as input.
    These layers should belong to keras.layers
    '''
    def add_identity_block(self, layers):

        X_shortcut = self.X
        for layer in layers:
            self.X = layer(self.X)
        self.X = tf.keras.layers.Add()([self.X, X_shortcut])
        self.X = self.activation(self.X)


    '''
    ADD_LAYERS
    Adds layers to the model. Accepts a list of layers as input.
    '''
    def add(self, layers):
        for layer in layers:
            self.X = layer(self.X)
        self.X = self.activation(self.X)

    '''
    returns a keras model which needs to be compiled, fit and tested similar to any inbuilt keras model.
    '''
    def build_model(self):
        self.model = tf.keras.models.Model(inputs=self.X_input,  outputs=self.X, name='ResNet')
        return self.model
