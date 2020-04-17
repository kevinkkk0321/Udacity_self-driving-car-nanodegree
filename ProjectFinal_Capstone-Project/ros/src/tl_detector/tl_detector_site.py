#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier_site import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.xycoords_orig_waypoints = None
        self.kdtree_orig_waypoints = None
        
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.bridge = CvBridge()
        self.light_classifier_site = TLClassifier()
        self.listener = tf.TransformListener()
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.xycoords_orig_waypoints:
            self.xycoords_orig_waypoints = [[waypoint.pose.pose.position.x, 
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.kdtree_orig_waypoints = KDTree(self.xycoords_orig_waypoints)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        
        state = TrafficLight.UNKNOWN
        
        if self.pose and self.kdtree_orig_waypoints:
            light_wp, state = self.process_traffic_lights()
        
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            #rospy.loginfo("Red light at " + str(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            #rospy.loginfo("published previous signal")
        self.state_count += 1

    def get_closest_waypoint(self, xcoord, ycoord):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        id_closest_waypoint = self.kdtree_orig_waypoints.query([xcoord, ycoord], 1)[1]
        return id_closest_waypoint

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
        #    self.prev_light_loc = None
            return False
        
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        state = self.light_classifier_site.get_classification(cv_image)
        
        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_traffic_light = None
        
        if(self.pose):
            xpos = self.pose.pose.position.x
            ypos = self.pose.pose.position.y
            id_wp_car_position = self.get_closest_waypoint(xpos, ypos)

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        #min_delta_wpid_car_line = len(self.waypoints.waypoints)
        min_delta_wpid_car_line = 50

        for i, light in enumerate(self.lights):
            stop_line_xycoord = stop_line_positions[i]
            id_wp_stop_line_i = self.get_closest_waypoint(stop_line_xycoord[0], stop_line_xycoord[1])
            delta_wpid_car_line = id_wp_stop_line_i - id_wp_car_position
            if delta_wpid_car_line >= 0 and delta_wpid_car_line < min_delta_wpid_car_line:
                id_wp_closest_stop_line = id_wp_stop_line_i
                min_delta_wpid_car_line = delta_wpid_car_line
                closest_traffic_light = light

        if closest_traffic_light:
            state = self.get_light_state(closest_traffic_light)
            return id_wp_closest_stop_line, state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
