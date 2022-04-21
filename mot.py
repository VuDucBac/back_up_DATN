import torch
import os
import os.path as osp
from loguru import logger


import sys
sys.path.append('.')
sys.path.append('/home/cv/ByteTrack-main/mot/utils')
sys.path.append('/home/cv/ByteTrack-main/mot/tracking_utils')
sys.path.append('/home/cv/ByteTrack-main/mot')

from .detector import YoloX_detector
from .tracker.byte_tracker import BYTETracker

from .utils import get_model_info
from .tracking_utils.timer import Timer
from .utils.visualize import plot_tracking
from vehicle_counting import Vehicle_counting

class Multiple_object_tracking:
    def __init__(
        self,
        exp,
        args
        ):
        """
        Top level module that integrates detection
        and tracking together.
        It contains Detector as the first block
        Tracker is the second, which takes Detector's outputs as inputs
        """
        self.exp = exp
        self.args = args      
        
        if not self.args.experiment_name:
            self.args.experiment_name = self.exp.exp_name
        
        # create an output directory for storing outputs and trt model
        output_dir = osp.join("mot_outputs", self.args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # trt device (cpu/gpu)
        if self.args.trt:
            self.args.device = "gpu"
        self.args.device = torch.device("cuda" if self.args.device == "gpu" else "cpu")

        # take conf, nms thres, test size of exp from args
        if self.args.conf is not None:
            self.exp.test_conf = self.args.conf
        if self.args.nms is not None:
            self.exp.nmsthre = self.args.nms
        if self.args.tsize is not None:
            self.exp.test_size = (self.args.tsize, self.args.tsize)

        # get our model
        model = self.exp.get_model().to(self.args.device)
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.eval()

        if not self.args.trt:
            if self.args.ckpt is None:
                ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt
            logger.info("loading checkpoint")
            ckpt_load = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt_load["model"])
            logger.info("loaded checkpoint done.")
        
        # fuse model
        if self.args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if self.args.fp16:
            model = model.half()  # to FP16

        # if use trt
        if self.args.trt:
            assert not self.args.fuse, "TensorRT model is not support model fusing!"
            trt_file = osp.join(output_dir, "model_trt.pth")
            assert osp.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None
            
        # for counting frame id
        self.frame_id = 0
        
        # results of mot as output MOT Challenge format results (e.g. MOT20-01.txt)
        self.results = []
        
        # init detector and tracker, you can change type of tracker and detector here
        logger.info("Loading detector ...")
        self.detector = YoloX_detector(model, self.exp, trt_file, decoder, self.args.device, self.args.fp16)
        logger.info("Loading tracker ...")
        self.tracker = BYTETracker(self.args, frame_rate=30)

        # 
        self.object_counter = Vehicle_counting(tlwh=0,tid=0)
        
        # a timer for time measurement
        self.system_timer = Timer()
        self.detector_timer = Timer()
        self.tracker_timer = Timer()
        

    def run_mot(self, frame):
        """
        method to run Multiple object tracking
        this is called in application level (app.py file)
        
        * parameter:
        ------------
        un-processed frame, original frames taken out from video, img, stream, ...
        
        * return:
        ---------
        processed frame, with bounding boxes drawn and important info written as text, ...
        """
            
        # logger each 50 frames
        if self.frame_id % 50 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(self.frame_id, 1. / max(1e-5, self.system_timer.average_time)))
        
        # start measuring time
        self.system_timer.tic()
        
        self.detector_timer.tic()
        
        # doing detection job
        outputs, img_info = self.detector.inference(frame)
        #np.set_printoptions(precision=1,suppress=True) #nicely format the output
        #print(f'outputs from detector: {outputs[0]}, shape: {len(outputs)}, element shape: {len(outputs[0])}, type: {type(outputs)}\n\n\n\n\n')
        #print(f'outputs[0] from detector: {outputs[0][0]}, shape: {len(outputs[0])}, element shape: {len(outputs[0][0])}, type: {type(outputs[0])}\n\n\n\n\n')
        
        # stop measuring time
        self.detector_timer.toc()
        
        self.tracker_timer.tic()

        # goi ham de ve duong tren frame dau tien 
        self.object_counter.Set_line(self.frame_id, frame)

        # perform tracking job
        if outputs[0] is not None:
            online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    self.results.append(
                        f"{self.frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
                # execute counting
                self.object_counter.counting(tlwh,tid)
                    
            self.tracker_timer.toc()
            self.system_timer.toc()

            # draw line and text for counting
            self.object_counter.draw_line(frame)


            # plot the image with detection and tracking results
            processed_frame = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id = self.frame_id + 1, fps = 1. / self.system_timer.average_time)
        else:
            self.tracker_timer.toc()
            self.system_timer.toc()
            processed_frame = img_info['raw_img']
            
        self.frame_id += 1
        
        return processed_frame

        
    def print_timing_info(self,detector_time,tracker_time,sys_time):
        """
        method to print out timing info, great for evaluation of system
        this is called in application level (app.py file)
        
        * parameter:
        ------------
        detector time, tracker time, whole system time
        """
        # Logger into console
        logger.info('=========================Timing Stats=========================')
        logger.info(f"{'Avg detecting time:':<20}{1000*detector_time:>6.3f} ms = {(100*detector_time/sys_time):6.3f} % total time")
        logger.info(f"{'Avg tracking time:':<20}{1000*tracker_time:>6.3f} ms = {(100*tracker_time/sys_time):6.3f} % total time")
        logger.info('----------------------------Sum Up----------------------------')
        logger.info(f"{'Avg total system time:':<20}{1000*sys_time:>6.3f} ms = 100 % total time")
        logger.info(f"{'Avg FPS:':<18}{1/sys_time:>6.3f} fps")
