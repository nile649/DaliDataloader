def dataiter(path,opt):

    if opt.typedataloader=='Multicuda':
        from .DaliDataLoaderMulti import dataCUDA # Batch of images
        return dataCUDA(path,opt)
    elif opt.typedataloader=='cuda':
        from .DaliDataLoader import dataCUDA # Single image normalize for int -255 =/ 255
        return dataCUDA(path,opt)
    elif opt.typedataloader=='CPU':
        from .DataLoader import dataCPU # Single image normalize for int -255 =/ 255
        return dataCPU(path,opt)
    else:
        from .DataLoader import dataPILCPU # Single image normalize for int -1 =/ 1
        return dataPILCPU(path,opt)