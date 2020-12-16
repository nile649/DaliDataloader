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