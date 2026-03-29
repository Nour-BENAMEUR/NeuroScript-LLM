import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from t5_generator import TextOnlyReportGenerator
from dataset_text_only_ import TextOnlyDataset, TextOnlyCollator
from trainer_text import Trainer  

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_dataloaders(train_config):
    # Datasets
    traindata = TextOnlyDataset(csv_path=r"E:\data_nour\train_all.csv")
    valdata = TextOnlyDataset(csv_path=r"E:\data_nour\val_all.csv")
    
    # Collators
    train_collator = TextOnlyCollator()
    val_collator = TextOnlyCollator()
    
    # DataLoaders 
    trainloader = DataLoader(
        traindata,
        batch_size=train_config['train_batch_size'],
        collate_fn=train_collator,
        shuffle=True,
        pin_memory=True,
        num_workers=2,  
        drop_last=True   
    )

    valloader = DataLoader(
        valdata,
        batch_size=train_config['eval_batch_size'],
        collate_fn=val_collator,
        shuffle=False,
        pin_memory=True,
        num_workers=2, 
    )

    return trainloader, valloader

def train_model(model, trainloader, valloader, config, output_path):
    model.cuda()
    
    trainer = Trainer()
    trainer.train(
        model=model,
        dataloader=trainloader,
        eval_dataloader=valloader,
        warmup_ratio=config['warmup'],
        epochs=config['num_epochs'],
        optimizer_params={'lr': config['lr']},
        output_path=output_path,
        weight_decay=config['weight_decay'],
        accumulation_steps=config['accumulation_steps'],
        use_amp=False, 
    )

def main():
    set_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    
    modality = "text"  # Définition claire de la modalité
    print(f"🚀 Starting {modality.upper()}-only ablation study")

    # Configuration identique à MRI-only
    train_config = {
        'num_epochs': 50,
        'use_amp': False,  
        'warmup': 0.1,
        'lr': 5e-5,  
        'weight_decay': 0.01,
        'train_batch_size': 5, 
        'eval_batch_size': 6,
        'accumulation_steps': 4,  
        'gradient_clip': 1.0,
        'eval_steps': 100,
        'lr_scheduler': 'linear',  
        'warmup_steps': 1000,                   
        'temperature': 0.7,            
        'repetition_penalty': 1.5,     
        'max_new_tokens': 300,
        'gradient_accumulation_steps': 2,
        'save_steps': 1000,
    }

    print(f"Loading datasets for {modality.upper()}-only...")
    trainloader, valloader = get_dataloaders(train_config)
    
    print(f"Initializing {modality.upper()}-only model...")
    model = TextOnlyReportGenerator(t5_model="google/flan-t5-base")
    
    output_path = f"E:/data_nour/checkpoints/ablation_{modality}_only"
    
    print(f"Starting training for {modality.upper()}-only...")
    train_model(model, trainloader, valloader, train_config, output_path)
    
    print(f"✅ {modality.upper()}-only training completed!")
    print(f"📁 Model saved in: {output_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    main()
