"""
Implementation of the paper - "Improving  Unsupervised  Defect  Segmentationby  Applying  Structural  Similarity  To  Autoencoders"
https://arxiv.org/pdf/1807.02011
"""

import tensorflow as tf
from collections import OrderedDict
import numpy as np
import pathlib
import random
from tensorflow.examples.tutorials.mnist import input_data
import cv2

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



                # upsampled_layer  = tf.keras.layers.UpSampling2D((output_shape[0],output_shape[1]),data_format=None,interpolation='nearest')
                # upsampled = (upsampled_layer)(prevLayerOut)
                output_s = output_shape
                kernel_value = tf.zeros((output_s[1],output_s[2],output_s[3],output_s[3]), dtype=tf.float32)
                kernel_value = kernel_value[0, 0, :, :].assign(tf.eye(output_s[2],output_s[3]))
                kernel = tf.constant(kernel_value)

                # do the un-pooling using conv2d_transpose
                unpool = tf.nn.conv2d_transpose(prevLayerOut,
                                        kernel,
                                        output_shape=(output_shape),
                                        strides=(1, _filter[0],_filter[1], 1),
                                        padding='VALID')

                kernel = tf.Variable(tf.truncated_normal(_filter, dtype=tf.float32,
                                                        stddev=1e-1), name='weight')

                conv = tf.nn.conv2d_transpose(unpool,output_shape=(output_shape), filter = kernel, strides = _strides,padding=padding)
                # print (scope)
                # print (biases)
                biases = tf.Variable(tf.constant(0.0, shape=_biases, dtype=tf.float32),
                                    trainable=True, name='bias')
                bias = tf.nn.bias_add(conv, biases,name="bias_add")
                # if activation : 
                conv1 = tf.nn.relu(bias, name="activation")
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

    def MultiScaleSSIM(self,img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                       k1=0.01, k2=0.03, weights=None):
      """Return the MS-SSIM score between `img1` and `img2`.
      This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
      Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
      similarity for image quality assessment" (2003).
      Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
      Author's MATLAB implementation:
      http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
      Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
          maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
          for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
          the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
          the original paper).
        weights: List of weights for each level; if none, use five levels and the
          weights from the original paper.
      Returns:
        MS-SSIM score between `img1` and `img2`.
      Raises:
        RuntimeError: If input images don't have the same shape or don't have four
          dimensions: [batch_size, height, width, depth].
      """
      print(img1.shape)
      if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
      if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

      # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
      weights = np.array(weights if weights else
                         [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
      levels = weights.size
      downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
      im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
      mssim = np.array([])
      mcs = np.array([])
      for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1, im2, max_val=max_val, filter_size=filter_size,
            filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [convolve(im, downsample_filter, mode='reflect')
                    for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
      return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
              (mssim[levels-1] ** weights[levels-1]))

                
    def loss_ssim(self,im1,im2):


                ksizes = [1, 11, 11, 1]
                strides = [1, 1, 1, 1]
                rates = [1, 1, 1, 1]


                
                patches1 = tf.extract_image_patches(
                        im1,
                        ksizes,
                        strides,
                        padding = 'SAME',
                        rates = rates,
                        name=None
                )

                patches2 = tf.extract_image_patches(
                        im2,
                        ksizes,
                        strides,
                        padding = 'SAME',
                        rates = rates,
                        name=None
                )
                shape = tf.shape(im1)


                patches1 = tf.reshape(patches1,shape=[shape[0],ksizes[1],ksizes[2],-1])
                patches2 = tf.reshape(patches2,shape=[shape[0],ksizes[1],ksizes[2],-1])
                patches1 = tf.transpose(patches1, perm=[0,3,1,2])
                patches1 = tf.expand_dims(
                                            patches1,
                                            axis=4,
                                            name=None,
                                            dim=None
                                        )
                patches2 = tf.transpose(patches2, perm=[0,3,1,2])
                patches2 = tf.expand_dims(
                                            patches2,
                                            axis=4,
                                            name=None,
                                            dim=None
                                        )
                # shape1 = tf.math.reduce_max(tf.reshape(patches1,[-1]))
                # shape2 = tf.math.reduce_max(tf.reshape(patches2,[-1]))
                # patches1 = (tf.squeeze(patches1, 0))
                # patches2 = (tf.squeeze(patches2, 0))


            
                ssim_scores = (tf.image.ssim(patches1,patches2,max_val = 1.0))
                ssim_residual = (tf.reshape(ssim_scores,shape = shape,name = "ssim_residual"))
                
                flattened_res  = tf.reshape(ssim_residual,[shape[0],-1])
                flattened_res_diss = tf.math.negative(
                                    flattened_res,
                                    name="str_dissim"
                                )

                total_loss = tf.reduce_mean(tf.reduce_sum(flattened_res,name="loss"))
                print("---------9-9-9--90-99-9-9-9-9-9-9-9-",total_loss)

                l1_loss = tf.losses.absolute_difference(
                im1,
                im2,
                weights=1.0,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES
                # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
                )

                return total_loss , l1_loss


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


        output  = tf.math.sigmoid(output,name = "final_Activation")

        


        #computing ssim loss
        ssim2 =tf.losses.sigmoid_cross_entropy(output,self.curr_graph.get_tensor_by_name("inp:0") )

        # ssim2= (ssim2*80)/100
        # l1_loss= (l1_loss*20)/100
        # print("nnnnnnnnnnnnnnnnnnnnnn",ssim2)

        # ssim2 = tf.image.ssim(reconstructedImage,self.curr_graph.get_tensor_by_name("inp:0") , max_val=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.00002,beta1=0.000005,name = 'optimizer').minimize(ssim2)

        return ssim2,optimizer,output

        
    def _parse_function(self,filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string,channels=1)
        # image_decoded= image_decoded/255
        print("..................",image_decoded)
        return image_decoded
    
    def train(self):

        loss,optimizer ,outputRe= self.get_decoder_model()
        init = tf.initialize_all_variables()
        with tf.Session() as sess:

    # Run the initializer
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)

                sess.run(init)

                # Training
                data_root2 = "/home/jbmai/defects/DefectDetection/aug"
                data_root = pathlib.Path(data_root2)
                all_image_paths = list(data_root.glob('*'))

                all_image_paths = [str(path) for path in all_image_paths]
                print("mmmmmmmmmmmmmmmmmmmmmmm",all_image_paths)
                all_image_paths = tf.constant(all_image_paths)
                # image_path = random.choice(all_image_paths)
                # print("--------------------------",image_path)
                # path = "../Images"

                # img_raw = tf.read_file(all_image_paths[0])
                # img_tensor = tf.image.decode_image(img_raw)


                dataset = tf.data.Dataset.from_tensor_slices((all_image_paths))
                # dataset = dataset.apply(tf.contrib.data.unbatch())
                dataset = dataset.map(self._parse_function)
                dataset = dataset.apply(tf.contrib.data.unbatch())

                dataset= dataset.batch(100)
                print("00000000000000000000000000000000",dataset)
                iterator = dataset.make_one_shot_iterator()
                next_image_batch = iterator.get_next()
                print("===================================================",next_image_batch)
                with tf.Session() as session:
                    img_value = session.run(next_image_batch)
                    # print("-------------------------",img_value)
                    img_value = (img_value)/255
                    print("-------------------------",img_value)

                    img_n =[]
                    for img in (img_value):

                        img_n.append((np.resize(cv2.resize(img,(128,128)),(128,128,1))))
                        # print(x.shape)
                        # img_n.a
                    # img =  cv2.cvtColor(img_value,cv2.COLOR_RGB2GRAY)
                    # cv2.imshow("mm",img_value[])
                    # cv2.waitKey(0)
                    # print(img_n.si)
                    img_n = np.asarray(img_n)

                    print(img_n)
                    # print(img_n.shape)

                    # cv2.waitKey(0)
                # print(img_tensor.shape)
                step = 1000
                for i in range(1, step+1):
                    # Prepare Data
                    # Get the next batch of MNIST data (only images are needed, not labels)

                    # batch_x = cv2.(img_value, [128, 128])
                    # batch_x = cv2.resize(img,(128,128))

                    # cv2.imshow("fuck",batch_x)
                    # cv2.waitKey(0)
                    # # batch_x = cv2.resize(batch_x,(128,128))
                    # batch_x = np.reshape(batch_x,(1,128,128,1))



                    # batch_x = 

                    # Run optimization op (backprop) and cost op (to get loss value)
                    saver = tf.compat.v1.train.Saver()
                    try: 
                        saver.restore(sess,tf.train.latest_checkpoint('tr/'))
                    except:
                        pass
                    l, m ,o= sess.run([optimizer, loss,outputRe], feed_dict={self.inputImage:img_n})
                    print (m.shape)

                   
                    if (i % 1 == 0 or i == 1):
                        print('Step %i: Minibatch Loss: %f' % (i, m))
                    if (step % 10 == 0):
                    # ave_path = saver.save(sess, "/tmp/model.ckpt")
                        saver.save(sess, 'tr/my-model'+str(step) + '.ckpt', global_step=step)
                        writer = tf.compat.v1.summary.FileWriter('tr/events', sess.graph)
                        print(o*255)
                        cv2.imwrite("reconstructed/img"+str(i)+".jpg",o[0,:,:,:]*255)





mayank = SSIMAutoEncoder()

mayank.train()



