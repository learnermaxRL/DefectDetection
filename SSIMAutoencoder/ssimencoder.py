 

"""
Implementation of the paper - "Improving  Unsupervised  Defect  Segmentationby  Applying  Structural  Similarity  To  Autoencoders"


https://arxiv.org/pdf/1807.02011



"""

import tensorflow as tf
from collections import OrderedDict
import numpy as np



class SSIMAutoEncoder:

    def __init__(self,inputshape =(None,128,128,1) ):

       self.inputShape = inputshape
       self.curr_graph = tf.get_default_graph()
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

    def conv_layer_decoder(self,scopeName,prevLayerOut,kernelSize,output_shape,padding = 'SAME',activation = True):

        
        _filter = [kernelSize[0][0],kernelSize[0][1],kernelSize[0][2],kernelSize[0][3]]
        _strides = [1,(kernelSize[1][1]),(kernelSize[1][0]),1]
        _biases = [_filter[2]]

        print(_filter)
        # print(_strides)
        print(_biases)
        print ("-------------------")
        print(output_shape)
        with tf.name_scope(scopeName+"Decoder") as scope:

                kernel = tf.Variable(tf.truncated_normal(_filter, dtype=tf.float32,
                                                        stddev=1e-1), name='weight')
                conv = tf.nn.conv2d_transpose(prevLayerOut,output_shape=output_shape, filter = kernel, strides = _strides,padding=padding)
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


    def conv_layer_encoder(self,scopeName,prevLayerOut,kernelSize,padding = 'SAME',activation = True):

        
        _filter = list(kernelSize[0])
        _strides = [1,(kernelSize[1][1]),(kernelSize[1][0]),1]
        _biases = [kernelSize[0][-1]]

        print(_filter)
        print(_strides)
        print(_biases)
        with tf.name_scope(scopeName+"Encoder") as scope:

                kernel = tf.Variable(tf.truncated_normal(_filter, dtype=tf.float32,
                                                        stddev=1e-1), name='weight')
                conv = tf.nn.conv2d(prevLayerOut, filter = kernel, strides = _strides,padding=padding)
                # print (scope)
                # print (biases)
                biases = tf.Variable(tf.constant(0.0, shape=_biases, dtype=tf.float32),
                                    trainable=True, name='bias')
                bias = tf.nn.bias_add(conv, biases,name = "lastOp")
                if activation : 
                    conv1 = tf.nn.relu(bias, name="lastOp")
                else:
                    conv1 = bias
                    return conv1

                # print_activations(conv1)
                return conv1
                


    def get_encoder_model(self):
        # print (self.kernelSizesEncdoer["conv1"])
        for i, (key) in enumerate(self.kernelSizesEncdoer):
            # print ("--------------------")
            value = self.kernelSizesEncdoer[key]

            activation = True
            padding = 'SAME'

            if (i==0) :
                output = self.inputImage

            if (i == (len(self.kernelSizesEncdoer)-1)):

                activation = False
                padding = 'SAME'
                

            output = self.conv_layer_encoder(key,output,value,padding,activation)

        return output


    def get_decoder_model(self):


        padding = 'SAME'
        final_layer_encoder = self.get_encoder_model()
        

        all_keys = list(reversed(self.kernelSizesEncdoer))
        
        for i, (key) in enumerate(reversed(self.kernelSizesEncdoer)):
            # print ("--------------------")
            value = self.kernelSizesEncdoer[key]

            activation = True
            padding = 'SAME'

            if (i==0) :
                output = self.curr_graph.get_tensor_by_name(key+"Encoder"+"/lastOp"+":0")
                

            if (i == (len(self.kernelSizesEncdoer)-1)):

                activation = False
                # padding = 'VALID'
            print(key+"Encoder"+"lastOp")
            # print ([n.name for n in tf.get_default_graph().as_graph_def().node])
    

            try:
                output_shape = tf.shape(self.curr_graph.get_tensor_by_name(all_keys[i+1]+"Encoder"+"/lastOp"+":0"))
            except:
                output_shape = tf.shape(self.curr_graph.get_tensor_by_name("inp:0"))
            # self.output_shape =output_shape
            print (output_shape)
            output = self.conv_layer_decoder(key,output,value,output_shape,padding,activation)

        reconstructedImage = output

        #computing ssim loss
        ssim2 = tf.image.ssim(reconstructedImage,self.curr_graph.get_tensor_by_name("inp:0") , max_val=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.00002,beta1=0.000005,name = 'optimizer').minimize(ssim)

        return ssim2,optimizer


    
    def train(self):

        loss,optimizer = self.get_decoder_model()
        init = tf.initialize_all_variables()
        with tf.Session() as sess:

    # Run the initializer
                sess.run(init)

                # Training
                for i in range(1, num_steps+1):
                    # Prepare Data
                    # Get the next batch of MNIST data (only images are needed, not labels)
                    batch_x, _ = mnist.train.next_batch(batch_size)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, l = sess.run([optimizer, loss], feed_dict={self.inputImage: batch_x})

                    # Display logs per step
                    if i % display_step == 0 or i == 1:
                        print('Step %i: Minibatch Loss: %f' % (i, l))






                





    
    
