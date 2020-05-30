from .detect import Detector, Detection
import numpy as np
import os
import tensorrt as trt
from xavier_demo_det_utils import *


class Centernet_tensorrt(Detector):
    def __init__(self, args):
        super(Centernet_tensorrt).__init__()
        self.conf_thres=args.vis_thres
        self.model_detection = detection_block(
            deploy_file=os.path.join(args.detector_model_path, 'DLASeg.prototxt'),
            model_file=os.path.join(args.detector_model_path, 'DLASeg.caffemodel'),
            engine_file=os.path.join(args.detector_model_path, 'DLASeg.engine'),
            input_shape=(3, 512, 512),
            output_name=['conv_blob53', 'conv_blob55', 'conv_blob57'],
            data_type=trt.float32,
            flag_fp16=True,
            max_workspace_size=1,
            max_batch_size=1,
            num_class=1,

            )
        self.model_detection.set_max_per_image(args.max_per_image)
        self.process_caffemodel = process_caffemodel(self.model_detection)
        self.model_detection.set_output(*self.process_caffemodel.get_outputs())
    def __call__(self, img):
        self.model_detection.do_inference(img)
        detections =  self.model_detection.posprocess_detection(img)
        result = detections[1][np.where(detections[1][:,4] > self.conf_thres)]
        return [Detection(line[:4], line[4]) for line in result]

