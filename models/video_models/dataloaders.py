import torch
import numpy as np
from models.video_models.preprocess import *
#from models.video_models.dataset import MyDataset, pad_packed_collate


def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
                                Normalize( 0.0,255.0 ),
                                RandomCrop(crop_size),
                                HorizontalFlip(0.5),
                                Normalize(mean, std) ])

    preprocessing['val'] = Compose([
                                Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),
                                Normalize(mean, std) ])

    preprocessing['test'] = preprocessing['val']

    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines()

    # create dataset object for each partition
    dsets = {partition: MyDataset(
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz'
                ) for partition in ['test']}
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=False,
                        num_workers=8,
                        worker_init_fn=np.random.seed(1))for x in ['test']}
    return dset_loaders

def get_train_data_loaders(args):
    preprocessing = get_preprocessing_pipelines()

    # create dataset object for each partition
    dsets = {partition: MyDataset(
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz'
                ) for partition in ['train']}
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=False,
                        num_workers=8,
                        worker_init_fn=np.random.seed(1))for x in ['train']}
    return dset_loaders