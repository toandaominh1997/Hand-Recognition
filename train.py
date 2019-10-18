import os 
# os.system('pip install -r requirements.txt')
import argparse 
import time 
from pathlib import Path
from learning import Learning
from datasets import vinDataset
from models import accuracy_score
from utils import load_yaml, init_seed
import importlib
import torch
from tqdm import tqdm_notebook as tqdm

def getattribute(config, name_package, *args, **kwargs):
    module = importlib.import_module(config[name_package]['PY'])
    module_class = getattr(module, config[name_package]['CLASS'])
    module_args = dict(config[name_package]['ARGS'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    package = module_class(*args, **module_args)
    return package

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--train_cfg', type=str, default='./configs/train.yaml', help='train config path')
    args = parser.parse_args()
    config_folder = Path(args.train_cfg.strip("/"))
    config = load_yaml(config_folder)
    init_seed(config['SEED'])

    image_datasets = {x: vinDataset(root_dir = config['ROOT_DIR'], file_name = config['FILE_NAME'], num_triplet = config['NUM_TRIPLET'], phase = x) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = config['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'valid']}

    model = getattribute(config = config, name_package = 'MODEL')
    criterion = getattribute(config = config, name_package = 'CRITERION')
    metric_ftns = [accuracy_score] 
    optimizer = getattribute(config = config, name_package= 'OPTIMIZER', params = model.parameters())
    scheduler = getattribute(config = config, name_package = 'SCHEDULER', optimizer = optimizer)
    device = config['DEVICE']
    num_epoch = config['NUM_EPOCH']
    gradient_clipping = config['GRADIENT_CLIPPING']
    gradient_accumulation_steps = config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = config['EARLY_STOPPING']
    validation_frequency = config['VALIDATION_FREQUENCY']
    saved_period = config['SAVED_PERIOD']
    checkpoint_dir = Path(config['CHECKPOINT_DIR'], type(model).__name__)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    resume_path = config['RESUME_PATH']
    learning = Learning(model=model,
                        criterion=criterion,
                        metric_ftns= metric_ftns,
                        optimizer=optimizer,
                        device=device,
                        num_epoch=num_epoch,
                        scheduler = scheduler,
                        grad_clipping = gradient_clipping,
                        grad_accumulation_steps = gradient_accumulation_steps,
                        early_stopping = early_stopping,
                        validation_frequency = validation_frequency,
                        save_period = saved_period,
                        checkpoint_dir = checkpoint_dir,
                        resume_path = resume_path)
    
    learning.train(tqdm(dataloaders['train']), tqdm(dataloaders['valid']))

if __name__ == "__main__":
    main()