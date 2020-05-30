import os
import cv2
import time
import math
import numpy as np
import tensorrt as trt
from xavier_demo_det_utils import *
from mot.detection import Centernet_tensorrt
import argparse


#
ind_capture_device = 1
size_cap = [1280, 720]  # width height
# cap = cv2.VideoCapture(0)  #rec_20191119_165440.avi
cap = cv2.VideoCapture('./test_video/M19040313550500131.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_cap[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_cap[1])


# detection parameters
# OUTPUT_NAME = ["conv_45", "conv_53", "conv_61", "conv_69"]
model_detection = detection_block(
    deploy_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.prototxt',
    model_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.caffemodel',
    engine_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.engine',
    input_shape=(3, 512, 512),
    output_name=['conv_blob53', 'conv_blob55', 'conv_blob57'],
    data_type=trt.float32,
    flag_fp16=True,
    max_workspace_size=1,
    max_batch_size=1,
    num_class=1,

    )
model_detection.set_max_per_image(max_per_image=20)

process_caffemodel = process_caffemodel(model_detection)
context_detection, h_input_detection, d_input_detection, h_output_detection, d_output_detection, stream_detection = process_caffemodel.get_outputs()
model_detection.set_output(context_detection, h_input_detection, d_input_detection, h_output_detection, d_output_detection, stream_detection)
parser = argparse.ArgumentParser()
parser.add_argument('-dw', '--detector_model_path', required=True, help='root path of object detector models')
parser.add_argument('-rw', '--recognizer_checkpoint', required=False,
                    help='checkpoint file of TSN action recognizer')
parser.add_argument('-i', '--video_path', default='', required=False,
                    help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
parser.add_argument('-o', '--save_video', default='', required=False,
                    help='Path to the output video file. Leave it blank to disable.')
parser.add_argument('-s', '--save_result', default='', required=False,
                    help='Path to the output track result file. Leave it blank to disable.')
parser.add_argument('--num_segments', default=4, help='set segments num for action part')
parser.add_argument('--vis_thres', default=0.3, help='threshold of detection')
parser.add_argument('--max_per_image', default=20, help='max objects in per image')

args = parser.parse_args()
detector = Centernet_tensorrt(args)

while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[80:, :640]
    start_time = time.time()
    detections=detector(frame)
    # model_detection.do_inference(frame)
    # output_box_detection = model_detection.posprocess_detection(frame)
    end_time = time.time()
    print('total', end_time - start_time)