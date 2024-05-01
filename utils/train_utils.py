import h5py
import torch
import numpy as np
from PIL import Image
import PIL
import os 
import random


seed_val=44
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



def linspace_filter(n_frames):
    linspace = torch.cat([torch.linspace(1, 0, steps=(n_frames+1)//2)[:-1], torch.tensor([0]), torch.linspace(0, 1, steps=(n_frames+1) // 2)[1:]])
    weights = linspace.unsqueeze(1)
    return weights


def read_h5_processed(startnum,endnum):
    with h5py.File(path+'New Files/Processed Data/concat_D1-1140A_PA_2022-01-07-07-38-42_video_trig_processed.h5', 'r') as hdf:
        ls = list(hdf.keys())
        #print('List of dataset in the file:\n', ls)
        dataset = hdf.get('1')
        #print(len(dataset))
        data =np.array(dataset[startnum:endnum])
        data = np.expand_dims(data, axis=3)
    return data



def save_checkpoint(checkpoint_dir, step, model, optimizer=None, scheduler=None, score=None, mode="min", save=True):
    """
    CODE MODIFIED FROM: https://github.com/sreyas-mohan/udvd/blob/main/utils/train_utils.py
    """
    assert mode == "min" or mode == "max"
    last_step = getattr(save_checkpoint, "last_step", -1)
    save_checkpoint.last_step = max(last_step, step)
    step_checkpoints = False
    default_score = float("inf") if mode == "min" else float("-inf")
    best_score = getattr(save_checkpoint, "best_score", default_score)
    if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
        save_checkpoint.best_step = step
        save_checkpoint.best_score = score

    # if not no_save and step % save_interval == 0:
    if save:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model = [model] if model is not None and not isinstance(model, list) else model
        optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
        scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
        state_dict = {
            "step": step,
            "score": score,
            "last_step": save_checkpoint.last_step,
            "best_step": save_checkpoint.best_step,
            "best_score": getattr(save_checkpoint, "best_score", None),
            "model": [m.state_dict() for m in model] if model is not None else None,
            "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
            "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
            # "args": argparse.Namespace(**{k: v for k, v in vars(args).items() if not callable(v)}),
        }

        if step_checkpoints:
            torch.save(state_dict, os.path.join(checkpoint_dir, "checkpoint{}.pt".format(step)))
        if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
            torch.save(model, os.path.join(checkpoint_dir, "checkpoint_best.pt"))
        if step > last_step:
            torch.save(model, os.path.join(checkpoint_dir, "checkpoint_last.pt"))
