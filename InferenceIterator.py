import itertools
import sys
sys.path.append("..")
import itertools
from data import *
from utils import *
from PIL import Image
import random
from tqdm import trange, tqdm_notebook # Please remove this when deploying
class CountDataIterator():
    def __init__(self,**kwargs):
        self.reinit = True
        self.n_iter = 0
        for key, value in kwargs.items():
              setattr(self, key, value)
                
        self.result_list = list() 
        self.result_numpy = None
    
    def __reintialize__(self):
        '''
        Re-initialize the dali iterator when the image breaks the pipline.
        ''' 
#         pdb.set_trace()
        _new_folder_ = list(itertools.chain.from_iterable(self.notcorrupted))
        if len(_new_folder_) >=self.arg.sub_size:
            print("Number of iteration : {}".format(len(_new_folder_)//self.arg.sub_size))
            return chunkIt(_new_folder_,self.arg.sub_size) 
        elif len(_new_folder_) <self.arg.sub_size and len(_new_folder_) >0:
            return chunkIt(_new_folder_,len(_new_folder_))
        else:
            self.reinit = False
            return None
            
    
    def __initialize__(self):
        '''
        Initialize the dali iterator for the first time.
        '''
        folder = [os.path.join(root, name)
                  for root, dirs, files in os.walk(self.Dir)
                  for name in files
                  if name.endswith((".jpg", ".jpeg",".JPG",".JPEG",'.png','.PNG')) and "face_crop" not in root]

        folder = self.__check__(folder)
        random.shuffle(folder)
        print("Number of iteration : {}".format(len(folder)//self.arg.sub_size))
        return chunkIt(folder,self.arg.sub_size)            

    def __check__(self,folder):
        '''
        Check for any corrupted image on the first run.
        '''
        for filename in folder:
            try:
                im = Image.open(filename)
                im.verify() #It is able to get few defects but not rest. It may cause break in pure dali.
            except: 
                folder.remove(filename)
        return folder
        
    
    def __main__(self,files):
        '''
        Get new files leaving the corrupted image.
        
        ********* Please remove tqdm_notebook ***************
        '''
        self.notcorrupted = []
        
        for i,file in enumerate(tqdm_notebook((files))):
            try:
                dataloader = dataiter(file,self.arg)
                for i in tqdm_notebook(range(len(file)),leave=False): # Small loop over the file.
                    try:
                        batch = dataloader.next() # inference batch_size = 1
                        img_,labl = [batch[0]['img']],[batch[0]['img_og'].squeeze(0)]
                        self.n_iter = self.n_iter + 1
                        '''
                        Inference code
                        
                        '''
#                         pdb.set_trace()
                        self.main(self,img_,labl,file[i])

                    except:
                        self.notcorrupted.append(file[i:])
                        break
            except:
                self.notcorrupted.append(file[1:])
            
    def __call__(self):
        _files_ = self.__initialize__() # chunk of files, 1-N, 1 is batch of n files
        self.__main__(_files_)
        while(self.reinit):
#             pdb.set_trace()
            _files_ = self.__reintialize__()
            if(_files_!=None):
                self.__main__(_files_)
            else:
                break
            
        
        
    
        
    
    