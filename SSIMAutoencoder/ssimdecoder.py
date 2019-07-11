 

"""
Implementation of the paper - "Improving  Unsupervised  Defect  Segmentationby  Applying  Structural  Similarity  To  Autoencoders"


https://arxiv.org/pdf/1807.02011



"""

import tensorflow as tf
from collections import OrderedDict
import numpy as np



class SSIMEcnoder:

    def __init__(self,inputshape =(None,1,1,100) ):

       self.inputShape = inputshape
       self.inputImage = tf.placeholder(tf.float32, shape=self.inputShape,name="inp")
       self.kernelSizesEncdoer = OrderedDict({



           "conv1" : [(4,4,self.inputShape[-1],32),   (2,2)],  #[[h,w,in_channels,out_channels],[stride_H,stride_w]]
           "conv2" : [(4,4,32,32),  (2,2)],
           "conv3" : [(3,3,32,32),  (1,1)],
           "conv4" : [(4,4,32,64),  (2,2)],
           "conv5" : [(3,3,64,64),  (1,1)],
           "conv6" : [(4,4,64,128), (2,2)],
           "conv7" : [(3,3,128,64), (1,1)],
           "conv8" : [(3,3,64,32),  (1,1)],
           "conv9" : [(8,8,32,100), (1,1)]
        #    "conv9" : [(8,8,32,100), (1,1)]


       })


    def conv_layer_decoder(self,scopeName,prevLayerOut,kernelSize,padding = 'SAME',activation = True):

        
        _filter = [kernelSize[0][0],kernelSize[0][1],kernelSize[0][3],kernelSize[0][2]]
        _strides = [1,(kernelSize[1][1]),(kernelSize[1][0]),1]
        _biases = [kernelSize[0][-2]]

        print(_filter)
        print(_strides)
        print(_biases)
        with tf.name_scope(scopeName) as scope:

                kernel = tf.Variable(tf.truncated_normal(_filter, dtype=tf.float32,
                                                        stddev=1e-1), name='weight')
                conv = tf.nn.conv2d_transpose(prevLayerOut, filter = kernel, strides = _strides,padding=padding)
                # print (scope)
                # print (biases)
                biases = tf.Variable(tf.constant(0.0, shape=_biases, dtype=tf.float32),
                                    trainable=True, name='bias')
                bias = tf.nn.bias_add(conv, biases)
                # if activation : 
                conv1 = tf.nn.relu(bias, name=scope)
                # else:
                #     conv1 = bias
                #     return conv1

                # print_activations(conv1)
                return conv1
                


    def get_model(self):
        # print (self.kernelSizesEncdoer["conv1"])
        for i, (key) in enumerate(reversed(self.kernelSizesEncdoer)):
            # print ("--------------------")
            value = self.kernelSizesEncdoer[key]

            activation = True
            padding = 'SAME'

            if (i==0) :
                output = self.inputImage

            if (i == (len(self.kernelSizesEncdoer)-1)):

                activation = False
                # padding = 'VALID'
                

            output = self.conv_layer_decoder(key,output,value,padding,activation)

        return output



enc = SSIMEcnoder()
mod = enc.get_model()
# print (mod)

shape = list(enc.inputShape)
shape[0] = 1
tensor = np.zeros((1,1,1,100))
# print (tensor)

with tf.Session() as sess:
    
    writer = tf.summary.FileWriter("output", sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)
    l = sess.run(mod, feed_dict={enc.inputImage: tensor})
    print(l.shape)
    writer.close()
                





    
    
