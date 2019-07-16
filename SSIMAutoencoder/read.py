import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("../nn/image1.jpg",0)/255
img = cv2.resize(img,(128,128))
img = np.expand_dims(img,axis=2)
img = np.expand_dims(img,axis=0)
print (img.shape)
meta_path = './train/my-model.ckpt-1000.meta' # Your .meta file
# output_node_names = ['ReLu']
saver = tf.train.import_meta_graph(meta_path)
# saver = tf.compat.v1.train.Saver()

with tf.Session() as sess:
	# Restore the graph
	

	# Load weights
	saver.restore(sess,tf.train.latest_checkpoint('train/'))

	# Freeze the graph
	#for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "conv1Decoder"):
		#print(i)
	out = sess.run(tf.get_default_graph().get_tensor_by_name("final_Activation:0"),feed_dict={tf.get_default_graph().get_tensor_by_name("inp:0"):img})
	
	print (out.shape)
	out = out[0,:,:,:]*255
	
	print (out)
	cv2.imwrite("ssss2.jpg",out)
	#print (tf.get_default_graph().get_tensor_by_name("final_Activation:0"))
	#output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

	#print (output_node_names)
		# print (i)
	# Output nodes

	# print("------------------------",output_node_names)
	# frozen_graph_def = tf.graph_util.convert_variables_to_constants(
	# 	sess,
	# 	sess.graph_def,
	# 	[output_node_names[-1]])

	# # Save the frozen graph
	# with open('output_graph.pb', 'wb') as f:
	#   f.write(frozen_graph_def.SerializeToString())


# with tf.Session


# ssim_residual = tf.math.sigmoid(ssim_residual,name="ssim_residual_activated")
#                 dissim = tf.math.subtract(
                                        
#                                         tf.constant(1.0),
#                                         ssim_residual,
#                                         name=None
#                                     )


                
#                 flattened_res  = tf.reshape(dissim,[shape[0],-1])
