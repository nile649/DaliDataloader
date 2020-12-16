# DataLoader
CPU based dataloader using PIL and CV2.

# DaliDataloader
Custom modules for Dali Dataloader.

# DaliDataLoaderMulti
Example to load multiple files with different size.
** Causes memory leak.

# DaliDataLoader
Example to load files with batch of 1.

We didn't see much speed loss between Multi and Single data loader.

# InferenceIterator
Images are not perfect, and could be incomplete or corrupt which may or maynot get detected by PIL loader. This will break the pipeline of the DaliDataLoader in production, so the work around is initialize and re-initialize mulitple dataloader with small window of images.

Code is self explanatory by reading the comments. 

We did benchmark on 9.2k images.
```
PIL : 40min.
CV2 : 45-47min.
Dali : 6~10min. [Batch 1].
Dali : 6~10min. [Batch 64].
Dali : 6~20min. [Multiple Dali iterators with batch 1 and window of 100image-5image].
```

# Example Code of using InferenceIterator

modify variables or make small changes in the *DatLoader.py

```
from data import *
import timeit

class opt:
    def __init__(self):
        self.device = 'cuda'
        self.typedataloader = 'cuda' # arg to accomodate which dataloader will be used Multicuda > 1 && cuda == 1 && cpu >=1 
        self.batch_size = 1
        self.sub_size = 100  # Window of 100 images.
        self.num_workers = 4
        self.dataFolder = '/data/sample/'

arg = opt()

def _mainExecute_Detection_(obj,img_,labl,file):
    """Main code"""

dirpath = arg.dataFolder

kwargs = {
    "Dir":dirpath,
    "arg":arg,
    "main" : _mainExecute_Detection_,
    #"model" : model, MachineLearning model if any.

}
datacount = CountDataIterator(**kwargs)
    

```

# Function to chunk files to small windows.
```
def chunkIt(seq, batch_size):
    avg = batch_size
    out = []
    last = 0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

```