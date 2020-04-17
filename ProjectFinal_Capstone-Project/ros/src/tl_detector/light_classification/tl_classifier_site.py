from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
import numpy as np


class TLClassifier(object):
    def __init__(self):
        self.trained_model_folder = 'light_classification/model_site_02/'
        self.trained_model_graph = self.trained_model_folder + 'traffic_lights.meta'
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.saver = tf.train.import_meta_graph(self.trained_model_graph)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.trained_model_folder))
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("X:0")
        self.keep_prob = self.graph.get_tensor_by_name("Keep_prob:0")
        self.logits = self.graph.get_tensor_by_name("Logits:0")
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implements light color prediction
        test_prediction = 3
        req_width = 200
        req_height = 150
        resized_image = cv2.resize(image, (req_width, req_height), interpolation = cv2.INTER_AREA)
        norm_image = (resized_image - 127.5)/255.0
        test_input = np.expand_dims(norm_image, axis=0)
       
        test_prediction = self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.x: test_input, self.keep_prob: 1.0})
   
        if test_prediction == 0:
            #print("RED")
            return TrafficLight.RED
        elif test_prediction == 1:
            #print("YELLOW")
            return TrafficLight.YELLOW
        elif test_prediction == 2:
            #print("GREEN")
            return TrafficLight.GREEN
        else:
            #print("UNKNOWN")
            return TrafficLight.UNKNOWN

