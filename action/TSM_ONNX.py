from .MobileNet_TSM import MobileNetV2
import numpy as np
import os
import torch
from typing import Tuple
import tvm
import tvm.relay
import onnx
import tvm.contrib.graph_runtime as graph_runtime
import torch.onnx
import io
from .transforms import GroupNormalize, ToTorchFormatTensor
import torchvision


class TSM(object):
    def __init__(self, args):
        self.lib_fname = os.path.join(args.action_path, 'mobilenet_tsm_tvm_cuda.tar')
        self.graph_fname = os.path.join(args.action_path, 'mobilenet_tsm_tvm_cuda.json')
        self.params_fname = os.path.join(args.action_path, 'mobilenet_tsm_tvm_cuda.params')
        self.executor, self.ctx = self.torch2executor()
        self.HISTORY_LOGIT=args.if_action_history
        self.max_hist_len=args.max_hist_len
        self.predict_segments=args.predict_segments
        self.buffer = (
            tvm.nd.empty((1, 3, 8, 8), ctx=self.ctx),
            tvm.nd.empty((1, 4, 4, 4), ctx=self.ctx),
            tvm.nd.empty((1, 4, 4, 4), ctx=self.ctx),
            tvm.nd.empty((1, 8, 2, 2), ctx=self.ctx),
            tvm.nd.empty((1, 8, 2, 2), ctx=self.ctx),
            tvm.nd.empty((1, 8, 2, 2), ctx=self.ctx),
            tvm.nd.empty((1, 12, 2, 2), ctx=self.ctx),
            tvm.nd.empty((1, 12, 2, 2), ctx=self.ctx),
            # tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
            # tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
        )
        self.active_ids={}
    def torch2executor(self):
        print('-------------------loading recognizer engine-------------------')
        with open(self.graph_fname, 'rt') as f:
            graph = f.read()
        tvm_module = tvm.module.load(self.lib_fname)
        params = tvm.relay.load_param_dict(bytearray(open(self.params_fname, 'rb').read()))

        ctx = tvm.gpu()
        graph_module = graph_runtime.create(graph, tvm_module, ctx)
        for pname, pvalue in params.items():
            graph_module.set_input(pname, pvalue)

        def executor(inputs: Tuple[tvm.nd.NDArray]):
            for index, value in enumerate(inputs):
                graph_module.set_input(index, value)
            graph_module.run()
            return tuple(graph_module.get_output(index) for index in range(len(inputs)))

        return executor, ctx

    def frame_process(self, image):
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
        transform = torchvision.transforms.Compose([
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)])
        image_tensor = (transform(image))
        return image_tensor

    def process_output(self,idx_, history):
        # idx_: the output of current frame
        # history: a list containing the history of predictions



        # mask out illegal action
        # if idx_ in [7, 8, 21, 22, 3]:
        #     idx_ = history[-1]

        # use only single no action class
        # if idx_ == 0:
        #     idx_ = 2

        # history smoothing
        if idx_ != history[-1]:
            if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
                idx_ = history[-1]

        history.append(idx_)
        history = history[-self.max_hist_len:]

        return history[-1], history
    def __call__(self, objs):
        for obj in objs:
            if len(obj.feature_history)>2:
                if obj.id not in self.active_ids.keys():
                    self.active_ids[obj.id]=self.buffer
                img_tensor = self.frame_process(obj.feature_history[-1][1]['patch'])
                input_var = torch.autograd.Variable(img_tensor.view(1, 3, img_tensor.size(1), img_tensor.size(2)))
                img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=self.ctx)
                inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + self.active_ids[obj.id]
                outputs = self.executor(inputs)
                feat, self.active_ids[obj.id] = outputs[0], outputs[1:]
                if self.HISTORY_LOGIT:
                    obj.past_buffers.append(feat.asnumpy())
                    if len(obj.past_buffers)>=self.predict_segments:
                        obj.past_buffers =  obj.past_buffers[-(self.max_hist_len-self.predict_segments):]
                        avg_logit = sum( obj.past_buffers)

                        idx_ = np.argmax(avg_logit, axis=1)[0]
                    else:
                        idx_=None
                else:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                obj.action.append(idx_)
                # idx, obj.action = self.process_output(idx_, obj.action)
