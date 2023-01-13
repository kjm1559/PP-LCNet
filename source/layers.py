import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, DepthwiseConv2D, Flatten, SeparableConv2D
import tensorflow.keras.backend as K

def hard_swish(x):
    return x * (keras.activations.relu(x + 3., max_value=6.)/6.)

def hard_sigmoid(x):
    return tf.maximum(tf.zeros(tf.shape(x)), tf.math.minimum(tf.ones(tf.shape(x)), (x + 1)/2))

class SqueezeExcitation(keras.layers.Layer):
    def __init__(self, feature):
        super(SqueezeExcitation, self).__init__()
        self.gap = GlobalAveragePooling2D()
        self.squeeze = Dense(feature/4, activation='relu')
        self.excitation = Dense(feature, activation=hard_sigmoid)
    
    def call(self, x):
        y = self.excitation(self.squeeze(self.gap(x)))
        # scale
        y = x + x * tf.expand_dims(tf.expand_dims(y, axis=1), axis=1) 
        return y

class DepthSepCov(keras.layers.Layer):
    def __init__(self, feature, kernel_size, strides, padding='same', se_flag=False):
        super(DepthSepCov, self).__init__()
#         self.dw = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', activation=hard_swish)
#         self.pw = Conv2D(feature, kernel_size=(1, 1), strides=(1, 1), activation=hard_swish)
        self.sc = SeparableConv2D(feature, kernel_size=kernel_size, strides=strides, padding=padding, activation=hard_swish)
        # optional
        self.se_flag = se_flag
        if self.se_flag:
            self.se = SqueezeExcitation(feature)
        
    def call(self, x):
#         print(x.shape)
        y = self.sc(x)
#         y = self.pw(self.dw(x))
        if self.se_flag:
            y = self.se(y)
        return y
    
class SteamConv(keras.layers.Layer):
    def __init__(self):
        super(SteamConv, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation=hard_swish)
    
    def call(self, x):
        y = self.conv1(x)
        return y
    

class PP_LCNet(keras.layers.Layer):
    def __init__(self, feature):
        ''' original
        input                 224^2x3
        Conv2d          3x3 2 112^2x16
        depthSepConv    3x3 1 112^2x32
        depthSepConv    3x3 2 56^2x64
        depthSepConv    3x3 1 56^2x64
        depthSepConv    3x3 2 28^2x128
        depthSepConv    3x3 1 28^2x128
        depthSepConv    3x3 2 14^2x256
        depthSepConv*5  5x5 1 14^2x256
        depthSepConv    5x5 2 7^2x512(SE)
        depthSepConv    5x5 1 7^2x512(SE)
        GAP             7x7 1 1^2x512
        Conv2d, NBN     1x1 1 1^2x1280
        '''
        
        ''' simple version
        input                 28^2x1
        Conv2d          3x3 2 15^2x16
        depthSepConv    3x3 1 15^2x32
        depthSepConv    3x3 2 8^2x64
        depthSepConv*2  5x5 1 8^2x128
        depthSepConv    5x5 1 8^2x256(SE)
        depthSepConv    5x5 1 8^2x256(SE)
        GAP             7x7 1 1^2x256
        Conv2d, NBN     1x1 1 1^2x512
        
        ''' 
        super(PP_LCNet, self).__init__()
        self.steam = SteamConv()
        # 3x3
        self.dep1 = DepthSepCov(32, (3, 3), (1, 1))
        self.dep2 = DepthSepCov(64, (3, 3), (2, 2))
        
        # 5x5
        self.dep3 = DepthSepCov(128, (5, 5), (1, 1))
        self.dep4 = DepthSepCov(128, (5, 5), (1, 1))
        
        self.dep5 = DepthSepCov(256, (5, 5), (1, 1), se_flag=True)
        self.dep6 = DepthSepCov(256, (5, 5), (1, 1), se_flag=True)
        
        self.gap = GlobalAveragePooling2D()
        self.conv1 = Conv2D(512, kernel_size=(1, 1), activation=hard_swish, padding='same')
        self.fl = Flatten()
        self.fc1 = Dense(feature, activation='softmax')
        
    def call(self, x):
        y = self.steam(x)
        
        y = self.dep2(self.dep1(y))
        y = self.dep4(self.dep3(y))
        
        y = self.dep6(self.dep5(y))
        y = tf.expand_dims(tf.expand_dims(self.gap(y), axis=1), axis=1)
        y = self.fl(self.conv1(y))
        y = self.fc1(y)
        
        return y

class compare_CNN(keras.layers.Layer):
    def __init__(self, feature):
        super(compare_CNN, self).__init__()
        # steam
        self.conv1 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu')

        # 3x3
        self.dep1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')
        self.dep2 = Conv2D(64, kernel_size=(1, 1), strides=(2, 2), activation='relu')

        # 5x5
        self.dep3 = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')
        self.dep4 = Conv2D(128, kernel_size=(1, 1), activation='relu')

        self.dep5 = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same')
        self.dep6 = Conv2D(256, kernel_size=(1, 1), activation='relu')

        self.gap = GlobalAveragePooling2D()
        self.conv = Conv2D(512, kernel_size=(1, 1), activation='relu', padding='same')
        self.fl = Flatten()
        self.fc1 = Dense(feature, activation='softmax')

    def call(self, x):
        y = self.conv1(x)

        y = self.dep2(self.dep1(y))
        y = self.dep4(self.dep3(y))

#         y = self.dep6(self.dep5(y))
        y = self.dep6(y)
        y = tf.expand_dims(tf.expand_dims(self.gap(y), axis=1), axis=1)

        y = self.fl(self.conv(y))
        y = self.fc1(y)

        return y

def PP_LCNet_model(input_shape, output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = PP_LCNet(output_shape)(inputs)
    return tf.keras.Model(inputs, outputs, name='PP_LCNet')

def CNN_model(input_shape, output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = compare_CNN(output_shape)(inputs)
    return tf.keras.Model(inputs, outputs, name='PP_LCNet')


