"""
import torch
import os
import os.path as osp
from loguru import logger

import sys
sys.path.append('.')
sys.path.append('/home/minh/ByteTrack/mot/utils')
sys.path.append('/home/minh/ByteTrack/mot/tracking_utils')

from .detector import YoloX_detector
from .tracker.byte_tracker import BYTETracker

from .utils import get_model_info
from .tracking_utils.timer import Timer
from .utils.visualize import plot_tracking
"""
import math
import cv2
from collections import deque
#from mot import Multiple_object_tracking



class Vehicle_counting:
    def __init__(
        self,
        tlwh,
        tid       
        ):
        """
        Vehicle counting module that count the number of vehicles that:
        - Pass through a line, a box
        - Move up or down, left or right the line
        """
        #self.crop
        self.already_track_id = []
        self.memory = {}
        self.already_counted = deque(maxlen=50)
        self.to_right = []
        self.to_left = []
        self.total_counter = []
        self.up_count = []
        self.down_count = []

        #self.line = [[(700, 100), (1200, 400)],[(1200, 950), (1900, 850)],[(200,400),(350,900)]]
        self.line = []
        self.line_angle = 0

        self.click = 0
        self.draw = False
        self.x = 0
        self.y = 0
    """
    def tlwh_to_tlbr (self,a,b,c,d)
        tlbr = []
        tlbr[0] = a
        tlbr[1] = b
        tlbr[2] = a + c
        tlbr[3] = b + d
        return tlbr
    """
    #save first frame of video
    """
    def getFirstFrame(self, frame_id, frame):
        if frame_id == 0 :
            cv2.imwrite("./mot_outputs/First_frame/first_frame.jpg",frame)  # save frame as JPEG file
            image = cv2.imread("./mot_outputs/First_frame/first_frame.jpg")
            k = 0
            while k != 112:
                cv2.imshow("First Frame",image)
                k = cv2.waitKey(0)

            cv2.destroyAllWindows()

    """
    def Get_mouse_Rectangle(self, event, x, y, flags, frame) :
        if event == cv2.EVENT_LBUTTONDOWN :
            self.line.append([(0,0),(0,0)])
            self.line[self.click][0] = (x,y)
            self.draw = True
            print ("diem dau:{0}".format(self.line[self.click][0]))
            self.x = x
            self.y = y
        #elif event == cv2.EVENT_MOUSEMOVE :
        #    if self.draw == True :
        #        cv2.line(image, (self.x,self.y),(x,y), (255,0,255),2)
        elif event == cv2.EVENT_LBUTTONUP :
            print ("click couter = {0}".format(self.click))
            self.line[self.click][1] = (x,y)
            self.click+=1
            self.draw = False

            #print("point1: {0}" .format(self.point1))
            print("line: {0}" .format(self.line))
            cv2.rectangle(frame, (self.x,self.y),(x,y), (255,0,255),2)
            cv2.imshow("Set line window", frame)
            self.click+=1
            self.draw = False

            crop = frame[self.y:y , self.x:x]
            print("CROP: {0}" .format(crop))

            #print("point1: {0}" .format(self.point1))
            print("line: {0}" .format(self.line))
            cv2.rectangle(frame, (self.x,self.y),(x,y), (255,0,255),2)
            #cv2.imshow("Set line window", frame)
            cv2.imshow("Traffic Light", crop)

            
    def Get_mouse(self, event, x, y, flags, frame) :
        if event == cv2.EVENT_LBUTTONDOWN :
            self.line.append([(0,0),(0,0)])
            self.line[self.click][0] = (x,y)
            self.draw = True
            print ("diem dau:{0}".format(self.line[self.click][0]))
            self.x = x
            self.y = y
        #elif event == cv2.EVENT_MOUSEMOVE :
        #    if self.draw == True :
        #        cv2.line(image, (self.x,self.y),(x,y), (255,0,255),2)
        elif event == cv2.EVENT_LBUTTONUP :
            print ("click couter = {0}".format(self.click))
            self.line[self.click][1] = (x,y)
            self.click+=1
            self.draw = False

            #print("point1: {0}" .format(self.point1))
            print("line: {0}" .format(self.line))
            cv2.line(frame, (self.x,self.y),(x,y), (255,0,255),2)
            cv2.imshow("Set line window", frame)
            self.click+=1
            self.draw = False

            #print("point1: {0}" .format(self.point1))
            print("line: {0}" .format(self.line))
            cv2.line(frame, (self.x,self.y),(x,y), (255,0,255),2)
            cv2.imshow("Set line window", frame)


    def Set_line(self, frame_id, frame):
        if frame_id == 0 :
            """
            cv2.imwrite("./mot_outputs/First_frame/first_frame.jpg",frame)  # save frame as JPEG file
            image = cv2.imread("./mot_outputs/First_frame/first_frame.jpg")
            """
            cv2.namedWindow("Set line window")
            cv2.setMouseCallback("Set line window", self.Get_mouse_Rectangle,frame)

            k = 0
            while k != 112:
                cv2.imshow("Set line window",frame)
                k = cv2.waitKey(0)
            cv2.destroyAllWindows()


    # get midpoint of bbox
    def get_midpoint (self,tlwh):
        return tlwh[0]+tlwh[2]/2, tlwh[1]+tlwh[3]/2
    #get center of a line
    def get_center (self, point0, point1):
        return int ((point0[0]+point1[0])/2), int ((point0[1]+point1[1])/2)
    # functions to draw line and text
    def draw_line_horizontal(self, frame, point0, point1, i):
        cv2.line(frame, point0, point1, (0, 0, 255), 2)
        cv2.putText(frame, f"{i+1}. Total: {self.total_counter[i]} ({self.to_right[i]} right, {self.to_left[i]} left)", (int(0.02 * frame.shape[1]), int(0.1 * (i+1) * frame.shape[0])), 0,
                0.7e-3 * frame.shape[0], (0, 0, 255), 2)
        cv2.putText(frame, f"{i+1}", self.get_center(point0, point1), 0,
                0.5e-3 * frame.shape[0], (255, 255, 0), 1)
        #cv2.putText(frame, f"Line angle: {self._line_angle(point0, point1)} ", (int(0.7 * frame.shape[1]), int(0.15 * frame.shape[0])), 0,
        #       0.7e-3 * frame.shape[0], (0, 0, 255), 2)


    def draw_line_vertical(self, frame, point0, point1, i):
        cv2.line(frame, point0, point1, (0, 255, 255), 2)
        cv2.putText(frame, f"{i+1}. Total: {self.total_counter[i]} ({self.up_count[i]} up, {self.down_count[i]} down)", (int(0.02 * frame.shape[1]), int(0.1 * (i+1) * frame.shape[0])), 0,
               0.7e-3 * frame.shape[0], (0, 255, 255), 2)
        cv2.putText(frame, f"{i+1}", self.get_center(point0, point1), 0,
                0.5e-3 * frame.shape[0], (255, 255, 0), 1)
        #cv2.putText(frame, f"Line angle: {self._line_angle(point0, point1)} ", (int(0.05 * frame.shape[1]), int(0.15 * frame.shape[0])), 0,
        #        0.7e-3 * frame.shape[0], (0, 255, 255), 2)

    def draw_line(self, frame):
        for i in range(len(self.line)) :
            if self._line_angle(self.line[i][0],self.line[i][1]) <= 45 :
                self.draw_line_vertical(frame,self.line[i][0],self.line[i][1],i)
            if self._line_angle(self.line[i][0],self.line[i][1]) > 45 :
                self.draw_line_horizontal(frame,self.line[i][0],self.line[i][1],i)

    # functions to see if point A&B are at different side of line CD
    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)
    """
    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))
    """
    # get the angle of the line to decide is it vertical or horizontal
    def _line_angle(self,line0,line1 ):
        x = abs(line0[0]-line1[0])
        y = abs(line0[1]-line1[1])
        return math.degrees(math.atan2(y, x))
    # start counting
    def counting(self,tlwh,tid):
        """
        method to ...
        
        * parameter:
        ------------
        tlwh: bounding box thu duoc tu tracker trong mot
        tid: track id thu duoc tu tracker trong mot
        
        * return:
        up_count, down_count, total_count, to_right, to_left increse each time an object pass under conditions.
        ---------
        
        """

        for i in range(len(self.line)) :
            self.total_counter.append(0)
            self.down_count.append(0)
            self.up_count.append(0)
            self.to_left.append(0)
            self.to_right.append(0)

        bbox = tlwh
        midpoint = self.get_midpoint(bbox)


        if tid not in self.already_track_id:
            self.memory[tid] = deque(maxlen=2)
            self.already_track_id.append(tid)
            self.memory[tid].append(midpoint)

        previous_midpoint = self.memory[tid][0]
           
        for i in range(len(self.line)) :

            if self._intersect(midpoint, previous_midpoint, self.line[i][0], self.line[i][1]) and tid not in self.already_counted:

                self.total_counter[i] +=1
                self.already_counted.append(tid)  # Set already counted for ID to true.

                #angle = self._vector_angle(midpoint, previous_midpoint)
                self.line_angle = self._line_angle(self.line[i][0],self.line[i][1])
                #print(f"line angle:{line_angle}")

                if self.line_angle < 45 :
                    """       
                    if angle < 0:
                        self.up_count += 1
                    if angle > 0:
                        self.down_count += 1
                    """

                    if midpoint[1] - previous_midpoint[1] > 0 :
                        self.down_count[i] += 1
                    elif midpoint[1] - previous_midpoint[1] < 0 :
                        self.up_count[i] += 1
                if self.line_angle > 45 :
                    if midpoint[0] - previous_midpoint[0] > 0 :
                        self.to_right[i] += 1
                    elif midpoint[0] - previous_midpoint[0] < 0 :
                        self.to_left[i] += 1
                """ 
                if  len(self.total_counter) == i-1 or len(self.total_counter) == 0 :
                    self.total_counter.append(0)
                    self.down_count.append(0)
                    self.up_count.append(0)
                    self.to_left.append(0)
                    self.to_right.append(0)
                """

