import re
import src.transformer
from src.transformer.models import TransformerEncoder
# from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
# import logging
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import torch
from src.new_code.load_data import CustomDataset
from src.new_code.utils import read_json, print_now
import os

def is_float(string):
    try:
      float(string)
      return True
    except ValueError:
      return False

# Read hparams from the text file
def read_hparams_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        hparams = {}
        for line in lines:
            if len(line) < 2 or line.startswith("#"):
              continue
            #print(line)

            line = line.strip().split('#')[0]

            key, value = line.strip().split(': ')
            value = value.replace('"','')
            if value in ['True', 'False']:
              if value == 'True':
                value = True
              else:
                value = False
            elif value.isdigit():
              value = int(value)
            elif is_float(value):
              value = float(value)
            hparams[key] = value # float(value) if value.isdigit() else value
            #print(key, value)
        return hparams

def get_callbacks(ckpoint_dir, counter):
  if os.path.exists(ckpoint_dir) is False:
    os.mkdir(ckpoint_dir)
  callbacks = [
    ModelCheckpoint(
      dirpath=ckpoint_dir,#'projects/baseball/models/2010',
      filename=str(counter)+'model-{epoch:02d}',
      monitor='val_loss_combined',
      save_top_k=-1 
    )
  ]
  return callbacks

def get_train_val_dataloaders(dataset, batch_size, train_split=0.8, shuffle=True):
  total_samples = len(dataset)
  train_size = int(train_split * total_samples)
  val_size = total_samples - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  return (
    DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=71),
    DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=71)
  )

def subset(data, lim):
  for key in data:
    data[key] = data[key][:lim]
  # data["input_ids"] = data["input_ids"][:lim]
  # data["padding_mask"] = data["padding_mask"][:lim]
  # data["original_sequence"] = data["original_sequence"][:lim]
  # data["target_tokens"] = data["target_tokens"][:lim]
  # data["target_pos"] = data["target_pos"][:lim]
  # data["target_cls"] = data["target_cls"][:lim]
  return data

def pretrain(cfg):
  hparams_path = cfg['HPARAMS_PATH']#'src/new_code/regular_hparams.txt'
  ckpoint_dir = cfg['CHECKPOINT_DIR']
  mlm_dir = cfg['MLM_DIR']
  hparams = read_hparams_from_file(hparams_path)
  if 'RESUME_FROM_CHECKPOINT' in cfg:
    print_now(f"resuming training from checkpoint {cfg['RESUME_FROM_CHECKPOINT']}")
    model = TransformerEncoder.load_from_checkpoint(
      cfg['RESUME_FROM_CHECKPOINT'], 
      hparams=hparams
    )
  else:
    model = TransformerEncoder(hparams)
  
  mlm_paths = []
  for root, dirs, files in os.walk(mlm_dir):
    for file_path in files:
      mlm_paths.append(os.path.join(root, file_path))

  LIM = 1000
  val_dataloader = None
  batch_size = cfg['BATCH_SIZE']
  for epoch in range(cfg['MAX_EPOCHS']):
    for counter, mlm_path in enumerate(mlm_paths):
      if counter == 0:
        if val_dataloader is None:
          with open(mlm_path, 'rb') as f:
            dataset = pickle.load(f)
            dataset.data = subset(dataset.data, LIM)
          val_dataloader = DataLoader(dataset, batch_size=batch_size)
        continue
      with open(mlm_path, 'rb') as f:
        dataset = pickle.load(f)
        dataset.data = subset(dataset.data, LIM)
      callbacks = get_callbacks(ckpoint_dir, counter)
      trainer = Trainer(
        default_root_dir=ckpoint_dir,
        callbacks=callbacks,
        max_epochs=1,
        accelerator='gpu',
        devices=1
      )
      # Create a data loader
      # train_dataloader, val_dataloader = get_train_val_dataloaders(
      #   dataset=dataset,
      #   batch_size=batch_size,
      #   train_split=hparams['train_split']
      # )
      train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

      print_now("training and validation dataloaders are created")
      print_now(f"# of batches in training: {len(train_dataloader)}")
      print_now(f"# of batches in validation: {len(val_dataloader)}")
      trainer.fit(model, train_dataloader)
      trainer.validate(model, val_dataloader)

if __name__ == "__main__":
  torch.set_float32_matmul_precision("medium")
  CFG_PATH = sys.argv[1]
  print_now(CFG_PATH)
  cfg = read_json(CFG_PATH)
  pretrain(cfg)