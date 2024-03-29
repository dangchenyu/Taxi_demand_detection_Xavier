import cv2
import logging
import numpy as np
from .tracklet import Tracklet


class Tracker:
    def __init__(self, detector, encoders, matcher, predictor=None, last_frame=None, max_ttl=3, max_feature_history=30,
                 max_detection_history=3000, min_time_lived=0):
        self.detector = detector
        self.encoders = encoders
        self.matcher = matcher
        self.last_frame = last_frame
        self.predictor = predictor
        self.max_ttl = max_ttl
        self.max_feature_history = max_feature_history
        self.max_detection_history = max_detection_history
        self.min_time_lived = min_time_lived
        self.max_id = 0
        self.tracklets_active = []
        self.tracklets_finished = []
        self.frame_num = 0
        self.logger = logging.getLogger('MOT')

    def clear(self):
        self.max_id = 0
        self.tracklets_active = []
        self.tracklets_finished = []
        self.frame_num = 0

    def tick(self, img):
        """
        Detect, encode and match, following the tracking-by-detection paradigm.
        The tracker works online. For each new frame, the tracker ticks once.
        :param img: A 3D numpy array with shape (H, W, 3). The new frame in the sequence.
        """
        self.frame_num += 1

        # Prediction
        self.predict(img)

        # Detection
        detections = self.detector(img)
        # Encoding
        features = self.encode(detections, img)

        # Data Association
        row_ind, col_ind = self.matcher(self.tracklets_active, features)

        # Tracklet Update
        self.update(row_ind, col_ind, detections, features)

        self.logger.info(
            'Frame #{}: {} target(s) active, {} object(s) detected'.format(self.frame_num, len(self.tracklets_active),
                                                                           len(detections)))

    def encode(self, detections, img):
        """
        Encode detections using all encoders.
        :param detections: A list of Detection objects.
        :param img: The image ndarray.
        :return: A list of dicts, with features generated by encoders for each detection.
        """
        features = [{'box': detections[i].box} for i in range(len(detections))]
        for encoder in self.encoders:
            _features = encoder(detections, img)
            for i in range(len(detections)):
                features[i][encoder.name] = _features[i]
        return features

    def predict(self, img):
        """
        Predict target positions in the incoming frame.
        :param img: The image ndarray.
        """
        if self.predictor is not None:
            if self.predictor.__class__.__name__ is 'ECCPredictor':
                self.predictor(self.tracklets_active, img, self.last_frame)
                self.last_frame = img.copy()
            else:
                self.predictor(self.tracklets_active, img)

    def update(self, row_ind, col_ind, detections, detection_features):
        """
        Update the tracklets.
        *****************************************************
        Override this function for customized updating policy
        *****************************************************
        :param row_ind: A list of integers. Indices of the matched tracklets.
        :param col_ind: A list of integers. Indices of the matched detections.
        :param detection_boxes: A list of Detection objects.
        :param detection_features: The features of the detections. It can be any form you want.
        """
        # Update tracked tracklets' features
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(self.frame_num, detections[col_ind[i]],
                                                     detection_features[col_ind[i]])

        # Deal with unmatched tracklets
        tracklets_to_kill = []
        unmatched_tracklets = []
        for i in range(len(self.tracklets_active)):
            if i not in row_ind:
                if self.tracklets_active[i].fade():
                    tracklets_to_kill.append(self.tracklets_active[i])
                else:
                    unmatched_tracklets.append(self.tracklets_active[i])

        # Kill tracklets that are unmatched for a while
        for tracklet in tracklets_to_kill:
            self.kill_tracklet(tracklet)

        # Create new tracklets with unmatched detections
        for i in range(len(detection_features)):
            new_tracklets = []
            if i not in col_ind:
                new_tracklet = Tracklet(0, self.frame_num, detections[i], detection_features[i], max_ttl=self.max_ttl,
                                        max_feature_history=self.max_feature_history,
                                        max_detection_history=self.max_detection_history)
                new_tracklets.append(new_tracklet)
                self.add_tracklet(new_tracklet)
            if self.predictor is not None:
                self.predictor.initiate(new_tracklets)

    def terminate(self):
        """
        Terminate tracking and move all active tracklets to the finished ones.
        """
        for tracklet in self.tracklets_active:
            self.kill_tracklet(tracklet)

    def assignment_matrix(self, similarity_matrix):
        """
        Calculate assignment matrix using the matching algorithm. Only for debugging.
        :param similarity_matrix: A 2D numpy array. The similarity matrix.
        :return: A 2D numpy array with the same shape as the similarity matrix. The assignment matrix.
        """
        row_ind, col_ind = self.matcher(similarity_matrix)
        assignment_matrix = np.zeros([similarity_matrix.shape[0], similarity_matrix.shape[1]])
        for i in range(len(row_ind)):
            assignment_matrix[row_ind[i], col_ind[i]] = 1

        # For debugging, display similarity matrix and assignment matrix
        if similarity_matrix.shape[0] > 0:
            print('row_ind: ', row_ind)
            print('col_ind: ', col_ind)
            cv2.imshow('similarity', cv2.resize(similarity_matrix, (600, 600), interpolation=cv2.INTER_NEAREST))
            cv2.imshow('assignment', cv2.resize(assignment_matrix, (600, 600), interpolation=cv2.INTER_NEAREST))
        return assignment_matrix

    def add_tracklet(self, tracklet):
        """
        Add a tracklet to the active tracklets after giving it a new ID.
        :param tracklet: The tracklet to be added.
        """
        tracklet.id = self.max_id
        self.max_id += 1
        self.tracklets_active.append(tracklet)

    def kill_tracklet(self, tracklet):
        self.tracklets_active.remove(tracklet)
        if tracklet.time_lived >= self.min_time_lived:
            self.tracklets_finished.append(tracklet)
