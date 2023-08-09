import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from folder_locations import get_data_folder, get_results_folder
import ct_experiment_utils as ceu
from argparse import ArgumentParser
from datetime import timedelta
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import random
from msd_pytorch.msd_module import MSDModule
import skimage.io
import re
import albumentations as A
import cv2
import sys
        
class LightningMSDSegmentation(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.module = MSDModule(c_in=3, c_out=1, depth=200, width=1)
        self.bce_loss = nn.BCELoss()
        self.args = args
        
        self.save_hyperparameters()
        
    def forward(self, x):
        return torch.sigmoid(self.module(x))
        
    def calc_loss(self, batch, batch_idx):
        inp, target = batch
        outp = self.forward(inp)
        return self.bce_loss(outp, target)

    def training_step(self, batch, batch_idx):
        loss = self.calc_loss(batch, batch_idx)
        
        self.log('loss_training', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('step_loss_training', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):
        loss = self.calc_loss(batch, batch_idx)
        
        self.log('loss_validation', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('step_loss_validation', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.args.lr_decay_step_size,
            gamma=self.args.lr_decay_gamma
        )
        return [optimizer], [scheduler]
        
def train_model_lighting(model, args, train_ds, val_ds, experiment_folder, gradient_clip_val=0, num_workers=2):
    # For distributed data-parallel training, the effective batch size
    # will be args.batch_size * gpus * num_nodes.
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        num_workers=num_workers,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
        timeout=3600*3  #Timeout after loading data for 3 hours
    )

    if val_ds is not None:
        # For distributed data-parallel training, the effective batch size
        # will be args.batch_size * gpus * num_nodes.
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            num_workers=num_workers,
            shuffle=False,
            batch_size=args.batch_size
        )
    else:
        val_dl = None
    
    # add callbacks to save the model
    validation_callback = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "validation_loss",
        filename="best_validation_loss_{epoch}",
        monitor="loss_validation",
        save_top_k=1,
        mode="min")
    
    training_callback = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "training_loss",
        filename="best_training_loss_{epoch}",
        monitor="loss_training",
        save_top_k=1,
        mode="min")
    
    training_step_callback = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "training_step_loss",
        filename="best_training_step_loss_{step}",
        monitor="step_loss_training",
        save_top_k=1,
        every_n_train_steps=10,
        mode="min",
        save_last=True)
        
    epoch_callback = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "hour",
        filename="{epoch}_{step}",
        train_time_interval=timedelta(hours=1),
        save_top_k=-1)
        

    # most basic trainer, uses good defaults
    (experiment_folder / "tb_logs").mkdir(exist_ok=True)
    logger = TensorBoardLogger(experiment_folder, name='tb_logs')
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=DDPPlugin(find_unused_parameters=False),
        logger=logger,
        enable_checkpointing=True,
        callbacks=[validation_callback, training_callback, training_step_callback, epoch_callback],
        gradient_clip_val=gradient_clip_val
    )
    trainer.fit(model, train_dl, val_dl)
    
class AugmentedPhotoDataset(torch.utils.data.Dataset):

    def __init__(self, photos_path, masks_path, indices, photos_mean, photos_std, apply_augmentations=True):
        self.photos_mean = photos_mean
        self.photos_std = photos_std
        
        self.load_dataset(photos_path, masks_path, indices)
        
        if apply_augmentations:
            self.augmentations = A.Compose([
                A.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.005, always_apply=False, p=0.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                A.GaussNoise(var_limit=(0, (5/255)**2), p=0.5),
                A.Normalize(mean=self.photos_mean, std=self.photos_std, max_pixel_value=1.0),
                ])
        else:
            self.augmentations = A.Compose([
                A.Normalize(mean=self.photos_mean, std=self.photos_std, max_pixel_value=1.0),
                ])
            
    def load_dataset(self, photos_path, masks_path, indices):
        self.photos = []
        self.masks = []
        
        for i, index in enumerate(indices):
            apple, picture = index
            photo_path = photos_path / f"{apple}" / f"Kanzi{apple}_slice_{picture}.png"
            mask_path = masks_path / f"{apple}" / f"Kanzi{apple}_slice_{picture}.png"
            
            self.photos.append(skimage.io.imread(photo_path).astype(np.float32)/255)
            self.masks.append((255-skimage.io.imread(mask_path)[:,:,0]).astype(np.float32)/255)
        
    def __getitem__(self, index):
        aug_data = self.augmentations(image=self.photos[index], mask=self.masks[index])
        image = torch.from_numpy(np.moveaxis(aug_data["image"], 2, 0))
        mask = torch.from_numpy(aug_data["mask"][None, :, :])
        
        return image, mask

    def __len__(self):
        return len(self.photos)
    
def find_mask_indices(masks_path):
    picture_indices = []

    for i in range(1, 121):
        files = list((masks_path / f"{i}").glob("Kanzi*_slice_*.png"))
        for file in files:
            slice_number = int(re.findall('\d+', file.name)[-1])
            picture_indices.append((i, slice_number))

    return picture_indices
        
if __name__ == '__main__':
    scans_folder = get_data_folder()
    photos_path = scans_folder / "slice_photos_crop"
    masks_path = scans_folder / "slice_photo_masks_crop"
    save_folder = get_results_folder()
    experiment_folder = ceu.make_new_experiment_folder(save_folder, name="picture_slice_segmentation")
    
    split_apple_nrs = [
        [40, 34, 43, 9, 21, 91, 11, 24, 31, 47, 64, 74, 83, 99, 107],
        [58, 49, 50, 15, 25, 92, 12, 26, 32, 51, 66, 76, 85, 100, 113],
        [39, 69, 19, 82, 28, 96, 13, 27, 33, 54, 67, 77, 87, 101, 115],
        [88, 56, 97, 110, 37, 102, 17, 29, 42, 57, 70, 78, 90, 103, 116],
        [62, 68, 71, 63, 20, 81, 10, 22, 30, 44, 60, 73, 80, 95, 106, 119]
    ]
    split_nr = 0
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--learning_rate', default=1e-4/2, type=float)
    parser.add_argument('--lr_decay_step_size', default=150, type=int)
    parser.add_argument('--lr_decay_gamma', default=0.5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # parse params
    args = parser.parse_args()
    
    mask_indices = find_mask_indices(masks_path)
    val_indices = [index for index in mask_indices if index[0] in split_apple_nrs[split_nr]]
    train_indices = [index for index in mask_indices if index not in val_indices]
    #print("val_indices:")
    #print(len(val_indices))
    #print("train_indices:")
    #print(len(train_indices))
    
    train_ds = AugmentedPhotoDataset(photos_path, masks_path, train_indices, (0.65979585, 0.51474652, 0.34878069), (0.32747177, 0.28363168, 0.17863797))
    val_ds = AugmentedPhotoDataset(photos_path, masks_path, val_indices, (0.65979585, 0.51474652, 0.34878069), (0.32747177, 0.28363168, 0.17863797), apply_augmentations=False)
    
    #checkpoint_path = "/export/scratch3/des/experiments_kanzi_apple_browning/2022-12-09_picture_slice_segmentation_1/checkpoints/training_step_loss/last.ckpt"
    #model = LightningMSDSegmentation.load_from_checkpoint(checkpoint_path)
    
    model = LightningMSDSegmentation(args)
    
    train_model_lighting(model, args, train_ds, val_ds, experiment_folder, gradient_clip_val=None, num_workers=2)
