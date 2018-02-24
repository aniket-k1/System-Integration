#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3
CHEAT_TRAFFIC_LIGHTS  = 0       #to be able to operate without classifier
CAR_HALF_LENGTH_WP = 3          #to align car at traffic light
STOP_AREA_AHEAD_WP = 15         #in case car is not fully stooped, still try to stop
FREQ               = 15         #hertz, to control image frequency

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.init_completed = False
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl_poses = list()       # traffic light waypoint positions in Pose object
        self.tl_wp_indices = list()  # traffic light waypoint indices initialization to empty list

        self.simulator = True if rospy.get_param('~sim') == 0 else False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size = 1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size = 1)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size = 1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size = 1)


        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)


        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(simulator=self.simulator)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN       # state of traffic light
        self.last_state = TrafficLight.UNKNOWN  # last state of traffic light
        self.state_count = 0                    # consecutive occurance of same state
        self.last_wp = -1                       # last waypoint of tarffic light
        self.wp2light = -1                      # waypoints between car and traffic light
        self.states   = list()                  # used with /vehicle/traffic_lights
        self.init_completed = True
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        self.set_tl_wp_indices()  # precomputes waypoints of tarffic lights


    def traffic_cb(self, msg):
        self.lights = msg.lights

        if (self.lights):
            self.states = list()
            for tf in self.lights:
                self.states.append(tf.state)

        if(CHEAT_TRAFFIC_LIGHTS):
            self.image_cb(None)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        if (self.init_completed == False):
            return

        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        rate = rospy.Rate(FREQ)

        if self.state != state:                                     # if state changes
            self.state_count = 0                                    # reset counter
            self.state = state                                      # set state
        elif self.state_count >= STATE_COUNT_THRESHOLD:             # if confident
            self.last_state = self.state                            # set last_state
            light_wp = light_wp if state == TrafficLight.RED else -1# set light_wp only if red
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:                                                       # if not sure use use last_wp
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
        rate.sleep()


    def set_tl_wp_indices(self):
        # tl = traffic light
        # pre-maps each traffic light to a base waypoint

        if self.waypoints is None:
            return

        # traffic light positions
        tl_positions = self.config['stop_line_positions']

        self.tl_poses = list()       # traffic light waypoint positions in Pose object
        self.tl_wp_indices = list()  # traffic light waypoint indices initialization to empty list


        for tlp in tl_positions:
            tl_pose = Pose()
            tl_pose.position.x = tlp[0]   #convert tl_position to Pose
            tl_pose.position.y = tlp[1]   #convert tl_position to Pose
            self.tl_poses.append(tl_pose)
            temp = self.get_closest_waypoint(tl_pose)
            self.tl_wp_indices.append(temp)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        if self.waypoints is None:
            return -1

        min_dist  = 10000
        min_index = -1

        pos_x = pose.position.x
        pos_y = pose.position.y
        # check all the waypoints to see which one is the closest to our current position
        for i, waypoint in enumerate(self.waypoints):
            wp_x = waypoint.pose.pose.position.x
            wp_y = waypoint.pose.pose.position.y
            dist = math.sqrt((pos_x - wp_x) ** 2 + (pos_y - wp_y) ** 2)
            if (dist < min_dist):  # we found a closer wp
                min_index = i  # we store the index of the closest waypoint
                min_dist = dist  # we save the distance of the closest waypoint

        # returns the index of the closest waypoint
        if (min_index-CAR_HALF_LENGTH_WP >= 0):
            return min_index-CAR_HALF_LENGTH_WP
        else:
            return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #if light too far away do not bother
        if self.wp2light > 200:
            return 4

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #rospy.loginfo('get light')
        classification = self.light_classifier.get_classification(cv_image)
        return classification


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # if  car position is available then get nearest waypoint
        if (self.pose):
            car_wp = self.get_closest_waypoint(self.pose.pose)
        else:
            return -1, TrafficLight.UNKNOWN

        #TODO find the closest visible traffic light (if one exists)
        min_wp_distance = 100000  # large distance --- waypoint units
        light = None
        l_wp = -1  # so far nothing detected
        for i, tl_wp in enumerate(self.tl_wp_indices):  # for each waypoint index of traffic light
            wp_distance = tl_wp - car_wp  # distance between car and traffic light, wp=waypoint
            cond1 = wp_distance > 0  # light is in front
            cond2 = (wp_distance < min_wp_distance)  # new minimum found
            if (cond1 and cond2):  # choose wp_distance
                min_wp_distance = wp_distance
                l_wp = tl_wp
                light = self.tl_poses[i]

        if (l_wp == -1):
            self.wp2light = 10000   #large number to ignore image
        else:
            self.wp2light = l_wp - car_wp


        if light:
            if (CHEAT_TRAFFIC_LIGHTS and (cheat_state_index != -1)):
                state = self.states[cheat_state_index]
            else:
                state = self.get_light_state(light)
            #rospy.loginfo('process_tl: car_wp=%d, light_wp=%d, d=%d state=%d', car_wp, l_wp, self.wp2light, state)
            return l_wp, state
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
