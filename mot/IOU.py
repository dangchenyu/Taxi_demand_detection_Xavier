from .tracker import Tracker
from mot.metric import IoUMetric
from mot.encode import ImagePatchEncoder
from mot.associate import HungarianMatcher
from mot.predict import KalmanPredictor

class IoUTracker(Tracker):
    def __init__(self, detector, sigma_conf=0.4):
        print('-------------------loading Tracker-------------------')
        metric = IoUMetric(use_prediction=True)
        encoder = ImagePatchEncoder(resize_to=(32, 32))
        matcher = HungarianMatcher(metric, sigma=0.1)
        predictor = KalmanPredictor()
        super().__init__(detector, [encoder], matcher, predictor)
        self.sigma_conf = sigma_conf

