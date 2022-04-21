import argparse
import cv2
import time
import os
import os.path as osp

from loguru import logger
import mot
from mot.exp import get_exp
##
import sys
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from QtGui1 import Ui_MainWindow

##


def make_parser():
    
    # parser initialize
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    group = parser.add_mutually_exclusive_group()
    
    # input and output args
    required.add_argument('-i', '--input-uri', metavar="URI", required=True, help=
                          'URI to input stream, supported format as below:\n'
                          '1) image sequence (e.g. %%06d.jpg)\n'
                          '2) video file (e.g. file.mp4)\n'
                          '3) MIPI CSI camera (e.g. csi://0)\n'
                          '4) USB camera (e.g. /dev/video0)\n'
                          '5) RTSP stream (e.g. rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
                          '6) HTTP stream (e.g. http://<user>:<password>@<ip>:<port>/<path>)\n')
    optional.add_argument('-o', '--output-uri', metavar="URI", default='auto save', help=
                          'URI for local-saved output video file, please note:\n'
                          '1) if nothing is passed, auto save by deafault\n'
                          '   to \mot_outputs\(current time)\ \n'
                          '2) otherwise, url to a local folder should be passed\n')
    optional.add_argument('-sh', '--show', action='store_true', help='show visualizations of outputs')
    optional.add_argument('-str', '--stream', action='store_true', help='stream outputs to server') # developing feature
    optional.add_argument('--txt', metavar="FILE",
                          help='path to output MOT Challenge format results (e.g. MOT20-01.txt)')
    
    # other args
    optional.add_argument("-expn", "--experiment-name", type=str, default=None)
    optional.add_argument("-n", "--name", type=str, default=None, help="model name")

    # exp file args
    optional.add_argument("-f","--exp_file", default=None, type=str,
                        help="pls input your expriment description file")
    optional.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    optional.add_argument("--device", default="gpu", type=str,
                        help="device to run our model, can either be cpu or gpu")
    optional.add_argument("--conf", default=None, type=float, help="test conf")
    optional.add_argument("--nms", default=None, type=float, help="test nms threshold")
    optional.add_argument("--tsize", default=None, type=int, help="test img size")
    optional.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    optional.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision evaluating.")
    optional.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    optional.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    optional.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    optional.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    optional.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    optional.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    optional.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    optional.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    return parser




def main():
    # parse arguements
    args = make_parser().parse_args()
    
    # args detail
    logger.info("List of arguments detail: {}".format(args))
    
    # get experiment file detail
    exp = get_exp(args.exp_file, args.name)
    
    # init Multiple object tracking class, this class contains Detector and Tracker classes
    MOT = mot.Multiple_object_tracking(exp, args)
    
    # init txt, which is output MOT Challenge format results (e.g. MOT20-01.txt)
    
    txt = None
    """
    if args.txt is not None:
        Path(args.txt).parent.mkdir(parents=True, exist_ok=True)
        txt = open(args.txt, 'w')
    """
    # init show option and auto-save option
    if args.show:
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
    if args.output_uri == 'auto save':
        # auto save by deafault to \mot_outputs\(current time)\
        current_time = time.localtime()
        save_folder = osp.join("mot_outputs", time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        os.makedirs(save_folder, exist_ok=True)
        # create a output url with format: \(save_folder)\(name of input video)\
        args.output_uri = osp.join(save_folder, args.input_uri.split("/")[-1])
        

    # init video in out class
    resize_to = (1920, 1080)
    stream = mot.VideoIO(resize_to, args.input_uri, args.output_uri)
    
    # start capturing input 
    logger.info('Starting video capture...')
    stream.start_capture()
    
#def MOT():   
    # main part of Multiple object tracking
    try:
        # this is our main loop, loop until running out of frames, or KeyboardInterrupt
        while not args.show or cv2.getWindowProperty('Video', 0) >= 0:
            # in case input is http or rtsp stream
            if ("http://" in args.input_uri) or ("rtsp://" in args.input_uri):
                frame = stream.read_is_stream()
            # otherwise,...
            if not(("http://" in args.input_uri) or ("rtsp://" in args.input_uri)):
                frame = stream.read_not_stream()
                if frame is None:
                    logger.info('There is no frames left, terminating...')
                    break
            
            # run multiple objects tracking and return frames that be processed
            processed_frame = MOT.run_mot(frame)
            
            # write output MOT Challenge format results
            if txt is not None:
                txt.write(MOT.results[MOT.frame_id])
            
            # show results to a window
            if args.show:
                cv2.imshow('Processed Video', processed_frame)
                cv2.waitKey(0)
                
            # stream outputs to server
            if args.stream:
                stream.rtsp_stream(processed_frame)
            
            # local save outputs
            if args.output_uri is not None:
                stream.local_write(processed_frame)
                
    except KeyboardInterrupt:
        logger.info('\nUser interrupted, terminating...')
    finally:
        # clean up resources
        if txt is not None:
            txt.close()
        stream.release()
        cv2.destroyAllWindows()

    # print timing infomation (average time, fps)
    MOT.print_timing_info(MOT.detector_timer.average_time, MOT.tracker_timer.average_time, MOT.system_timer.average_time)

if __name__ == "__main__":
    main()
