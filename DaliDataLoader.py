from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from random import shuffle
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
import os
import pdb
class ExternalInputIterator(object):
    def __init__(self, batch_size=1, root_folder='', height=299, shuffle_files=False):
        self.root = root_folder
        self.batch_size = batch_size
        if type(root_folder)==list:
            self.files = root_folder
            self.list = True
        else:
            self.files = os.listdir(root_folder)#
            self.list = False

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch_src = []
        batch_targ = []
        for _ in range(self.batch_size):
            file = self.files[self.i]
            if self.list:
                f_src = open( file, 'rb')
            else:
                f_src = open(self.root+'/'+file, 'rb')
            batch_src.append(np.frombuffer(f_src.read(), dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch_src)
          
    next = __next__


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.data_iterator = data_iterator
        self.src = ops.ExternalSource()
#         self.targ = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB,)
        self.cast = ops.Cast(device = "gpu",
                             dtype = types.INT32)

        self.resize = ops.Resize(device="gpu", resize_x=750, resize_y=750, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[104.0, 117.0, 123.0],
                                            std=[1,1,1]
                                            )
        self.cmnp2 = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0, 0, 0],
                                            std=[255,255,255]
                                            ) 
    
    def define_graph(self):
        self.jpegs_src = self.src()
        images_src = self.decode(self.jpegs_src)
        return (self.cmnp(self.resize(images_src))),(self.cmnp2(images_src))#,scale,im_height, im_width#,self.cmnp(output_src_h),self.cmnp(output_src_v))
    def iter_setup(self):
        # the external data iterator is consumed here and fed as input to Pipeline
        src = self.data_iterator.next()
        self.feed_input(self.jpegs_src, src)


def dataCUDA(path: list(),opt):
    eii = ExternalInputIterator(batch_size=opt.batch_size, 
    root_folder=path, 
    height=opt.height)
    iterator = iter(eii)
    pipe = ExternalSourcePipeline(data_iterator=iterator, batch_size=opt.batch_size, num_threads=opt.num_workers, device_id=0)
    pipe.build()
#     print("DALI INITIATED")
    data_iter = DALIGenericIterator([pipe], ['img','img_og'],dynamic_shape=True,size=len(path), auto_reset=False)
    return data_iter

