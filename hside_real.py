import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
import datetime
import time
from hsi_setup import Engine, train_options, make_dataset

# Set wandb to offline mode for Kaggle or other environments without internet/login
if os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    os.environ["WANDB_MODE"] = 'offline'

if __name__ == '__main__':
    """Training settings"""
    

       
    

    parser = argparse.ArgumentParser(
    description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    
    img_options={}
    img_options['patch_size'] = 128

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    train_dir = '/kaggle/input/realistic-data/train'

    train_dataset = DataLoaderTrain(train_dir,50,img_options=img_options,use2d=engine.get_net().use_2dconv)
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
   
    print('==> Preparing data..')


    """Test-Dev"""
    
    basefolder = '/kaggle/input/realistic-data/test'
    
    mat_datasets = DataLoaderVal(basefolder, 50, None,use2d=engine.get_net().use_2dconv)
    
    mat_loader = DataLoader(
        mat_datasets,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    )      

    base_lr = opt.lr
    epoch_per_save = 20
    
    # Set initial learning rate, but adjust if resuming from a later epoch
    if opt.resume and engine.epoch >= 400:
        adjust_learning_rate(engine.optimizer, base_lr * 0.1)
        print(f'Adjusted learning rate to {base_lr * 0.1} for epoch {engine.epoch}')
    elif opt.resume and engine.epoch >= 200:
        adjust_learning_rate(engine.optimizer, base_lr * 0.5)
        print(f'Adjusted learning rate to {base_lr * 0.5} for epoch {engine.epoch}')
    else:
        adjust_learning_rate(engine.optimizer, opt.lr)
    
    print('loading finished')
    
    # Training loop using epochs from command line arguments
    # Don't reset epoch to 0 if resuming from checkpoint
    if not opt.resume:
        engine.epoch = 0
    
    print(f'Starting training from epoch {engine.epoch} to {opt.epochs}')
    while engine.epoch < opt.epochs:
        np.random.seed()

        if engine.epoch == 200:
        
            adjust_learning_rate(engine.optimizer, base_lr*0.5)
          
        if engine.epoch == 400:
            
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        
        engine.train(train_loader,mat_loader)
        
        engine.validate(mat_loader, 'real')
        

        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()
            
        # Increment epoch after saving checkpoint (so checkpoint reflects completed epoch)
        engine.epoch += 1
    
    # Safely finish wandb if it was initialized
    try:
        wandb.finish()
    except:
        pass
