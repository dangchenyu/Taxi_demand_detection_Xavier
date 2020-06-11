import os
import cv2
import time
import argparse
from detection import Centernet_tensorrt
from detection import Zebra_det
from action import TSM
from mot import IoUTracker
from utils import visualize_snapshot
from utils import get_video_writer

def track_and_recognize(tracker, recognizer,zb_detector, args):
    if args.video_path=='webcam':
        capture = cv2.VideoCapture(1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video_writer=get_video_writer(os.path.join(args.save_video, 'webcam.mp4'), 640, 640)
    else:
        capture = cv2.VideoCapture(args.video_path)
        video_base_name = os.path.split(args.video_path)[1]
        video_writer = get_video_writer(os.path.join(args.save_video, video_base_name), 640, 640)

    while True:
        walking_count = 0
        standing_count = 0
        start_time = time.time()
        ret, frame = capture.read()
        if tracker.frame_num < args.from_frame:
            continue
        if not ret:
            break
        if_zb_cross = zb_detector(frame)
        frame = frame[80:, :640]  # must be square
        tracker.tick(frame)
        frame = visualize_snapshot(frame, tracker)

        # Perform action recognition each second
        recognizer(tracker.tracklets_active)
        for tracklet in tracker.tracklets_active:
            try:
                if tracklet.action[-1] == 0:
                    box = tracklet.last_detection.box
                    frame = cv2.putText(frame, 'walking', (int(box[0] + 4), int(box[1]) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1)

                    walking_count+=1
                elif tracklet.action[-1] == 1:
                    box = tracklet.last_detection.box
                    frame = cv2.putText(frame, 'standing', (int(box[0] + 4), int(box[1]) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), thickness=1)

                    standing_count += 1
            except IndexError:
                continue
        end_time = time.time()
        cv2.putText(frame, 'FPS: {}'.format(int(1 / (end_time - start_time))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 78, 16), thickness=2)
        cv2.putText(frame, 'Walking: {}'.format(walking_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (139, 78, 16), thickness=2)
        cv2.putText(frame, 'Standing: {}'.format(standing_count), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 78, 16), thickness=2)
        cv2.putText(frame, 'Crossing: {}'.format(if_zb_cross), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 78, 16), thickness=2)

        print('FPS: {}  Walking:{} Standing: {}'.format(int(1 / (end_time - start_time)),walking_count, standing_count))
        cv2.imshow('Demo', frame)
        if args.if_save:
            video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dw', '--detector_model_path', default='./detection/caffe_models/',
                        help='root path of object detector models')
    parser.add_argument('--action_path', default='./action', help='root path of object detector models')
    parser.add_argument('-i', '--video_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('-o', '--save_video', default='./processed_videos/', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--num_segments', default=4, help='set segments num for action part')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='threshold of detection')
    parser.add_argument('--max_per_image', default=20, help='max objects in per image')
    parser.add_argument('--if_action_history', default=True, help='if combine past predictions in action recognition')
    parser.add_argument('--max_hist_len', default=10, help='max history length of action prediction')
    parser.add_argument('--predict_segments', default=4, help='smooth action segments')
    parser.add_argument('--if_save', default=False, type=bool, help='if save processed videos')
    parser.add_argument('--from_frame', default=0, help='from frame No')
    parser.add_argument('--zb_thres', default=1700, type=float, help='threshold of zebra crossing detection')
    args = parser.parse_args()

    recognizer = TSM(args)
    detector = Centernet_tensorrt(args)
    tracker = IoUTracker(detector, sigma_conf=0.3)
    zb_detector=Zebra_det(args.zb_thres)
    track_and_recognize(tracker, recognizer, zb_detector,args)