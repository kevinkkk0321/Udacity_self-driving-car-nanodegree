#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 
MAX_DECELERATION = 1

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        self.latest_pose = None
        self.orig_waypoints = None
        self.xycoords_orig_waypoints = None
        self.kdtree_orig_waypoints = None
        self.id_closest_waypoint = None
        self.id_traffic_waypoint = -1

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop_till_shutdown()

    def loop_till_shutdown(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.latest_pose and self.kdtree_orig_waypoints:
                self.id_closest_waypoint = self.find_id_closest_waypoint()
                self.id_farthest_waypoint = self.id_closest_waypoint + LOOKAHEAD_WPS
                final_lane = Lane()
                final_lane.header = self.orig_waypoints.header
                sliced_waypoints = self.orig_waypoints.waypoints[self.id_closest_waypoint : self.id_farthest_waypoint]
                if (self.id_traffic_waypoint) == -1 or (self.id_traffic_waypoint >= self.id_farthest_waypoint):
                    final_lane.waypoints = sliced_waypoints
                else:
                    final_lane.waypoints = self.braking_waypoints(sliced_waypoints)
                self.final_waypoints_pub.publish(final_lane)
            rate.sleep()

    def find_id_closest_waypoint(self):
        xpos = self.latest_pose.pose.position.x
        ypos = self.latest_pose.pose.position.y
        id_closest = self.kdtree_orig_waypoints.query([xpos, ypos], 1)[1]

        # check if the closest waypoint is ahead of or behind the current position
        current_posvec = np.array([xpos, ypos])
        closest_pt_posvec = np.array(self.xycoords_orig_waypoints[id_closest])
        prev_closest_pt_posvec = np.array(self.xycoords_orig_waypoints[id_closest-1])

        waypoint_vec = closest_pt_posvec - prev_closest_pt_posvec
        proposed_heading_vec = closest_pt_posvec - current_posvec

        if np.dot(waypoint_vec, proposed_heading_vec) <= 0:
            id_closest = (id_closest + 1) % len(self.xycoords_orig_waypoints)
        
        return id_closest   

    def pose_cb(self, msg):
        self.latest_pose = msg

    def waypoints_cb(self, waypoints):
        self.orig_waypoints = waypoints
        if not self.xycoords_orig_waypoints:
            self.xycoords_orig_waypoints = [[waypoint.pose.pose.position.x, 
                                           waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.kdtree_orig_waypoints = KDTree(self.xycoords_orig_waypoints)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. 
        self.id_traffic_waypoint = msg.data

    def obstacle_cb(self, msg):
        # Not required in the 2 term version of the Self-Driving Car Course
        pass

    def braking_waypoints(self, sliced_waypoints):
        updated_waypoints = []
        
        count_stopping_wp = max(self.id_traffic_waypoint - self.id_closest_waypoint - 3, 0)

        for i, waypoint in enumerate(sliced_waypoints):
            point = Waypoint()
            point.pose = waypoint.pose
            stopping_dist = self.distance(sliced_waypoints, i, count_stopping_wp)
            slowed_velocity = math.sqrt(2*MAX_DECELERATION*stopping_dist)
            if slowed_velocity < 1.0:
                slowed_velocity = 0.0
            point.twist.twist.linear.x = min(slowed_velocity, waypoint.twist.twist.linear.x)
            updated_waypoints.append(point)

        return updated_waypoints

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
