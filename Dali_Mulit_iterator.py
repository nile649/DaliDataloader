from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
import torch
import torch.utils.dlpack as torch_dlpack
import ctypes
import math
import numpy as np

to_torch_type = {
    np.dtype(np.float32) : torch.float32,
    np.dtype(np.float64) : torch.float64,
    np.dtype(np.float16) : torch.float16,
    np.dtype(np.uint8)   : torch.uint8,
    np.dtype(np.int8)    : torch.int8,
    np.dtype(np.int16)   : torch.int16,
    np.dtype(np.int32)   : torch.int32,
    np.dtype(np.int64)   : torch.int64
}
def feed_ndarray(dali_tensor, arr, cuda_stream = None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.
    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        dali_tensor.copy_to_external(c_type_pointer, None if cuda_stream is None else ctypes.c_void_p(cuda_stream))
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


class DALICOCOIterator(object):
    """
    COCO DALI iterator for pyTorch.
    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """
    def __init__(self, pipelines, size):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]

        self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = size
        self._pipes = pipelines

        # Build all pipelines
        for p in self._pipes:
            p.build()

        # Use double-buffering of data batches
        self._data_batches = [[None, None, None, None] for i in range(self._num_gpus)]
        self._counter = 0
        self._current_data_batch = 0
        self.output_map = ["image", "labels"]

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        self._first_batch = None
        self._first_batch = self.next()

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            raise StopIteration

        # Gather outputs
        outputs = []
        for p in self._pipes:
            p._prefetch()
        for p in self._pipes:
            outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            out_images = []
            bboxes = []
            labels = []
            # segregate outputs into image/labels/bboxes entries
            for j, out in enumerate(outputs[i]):
                if self.output_map[j] == "image":
                    out_images.append(out)
                elif self.output_map[j] == "labels":
                    labels.append(out)

            # Change DALI TensorLists into Tensors
            images = [x.as_tensor() for x in out_images]
            images_shape = [x.shape() for x in images]

            # Prepare labels shapes and offsets
            labels_shape = []
#             bbox_offsets = []

            torch.cuda.synchronize()
            for j in range(len(labels)):
#                 bbox_offsets.append([0])
                for k in range(len(labels[j])):
                    lshape = labels[j][k].shape()
#                     bbox_offsets[j].append(bbox_offsets[j][k] + lshape[0])
                    labels_shape.append(lshape)
#             pdb.set_trace()
            # We always need to alocate new memory as bboxes and labels varies in shape
            images_torch_type = to_torch_type[np.dtype(images[0].dtype())]
#             bboxes_torch_type = to_torch_type[np.dtype(bboxes[0][0].dtype())]
            labels_torch_type = to_torch_type[np.dtype(labels[0][0].dtype())]

            torch_gpu_device = torch.device('cuda', dev_id)
            torch_cpu_device = torch.device('cpu')

            pyt_images = [torch.zeros(shape, dtype=images_torch_type, device=torch_gpu_device) for shape in images_shape]
            pyt_labels = [torch.zeros(shape, dtype=labels_torch_type, device=torch_gpu_device) for shape in  labels_shape]
#             pyt_offsets = [torch.zeros(len(offset), dtype=torch.int32, device=torch_cpu_device) for offset in bbox_offsets]

            self._data_batches[i][self._current_data_batch] = (pyt_images, pyt_labels)

            # Copy data from DALI Tensors to torch tensors
            for j, i_arr in enumerate(images):
                feed_ndarray(i_arr, pyt_images[j])

            for j, l_list in enumerate(labels):
                for k in range(len(l_list)):
                    if (pyt_labels[k].shape[0] != 0):
                        feed_ndarray(l_list[k], pyt_labels[k])
                    pyt_labels[k] = pyt_labels[k].squeeze(dim=1)


        for p in self._pipes:
            p.release_outputs()
            p.schedule_run()

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self._num_gpus * self.batch_size
        return [db[copy_db_index] for db in self._data_batches]

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__();

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter > self._size:
            self._counter = self._counter % self._size
        else:
            logging.warning("DALI iterator does not support resetting while epoch is not finished. Ignoring...")