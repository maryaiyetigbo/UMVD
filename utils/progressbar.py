from collections import OrderedDict
from numbers import Number
from tqdm import tqdm
import torch


#CODE MODIFIED FROM: https://github.com/sreyas-mohan/udvd/blob/main/utils/progress_bar.py

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    def __init__(self, momentum=0.98):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class TimeMeter(object):
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)

    

class ProgressBar:
    """
    CODE MODIFIED FROM: https://github.com/sreyas-mohan/udvd/blob/main/utils/progress_bar.py
    """
    def __init__(self, iterable, epoch=None, prefix=None, quiet=False):
        self.epoch = epoch
        self.quiet = quiet
        self.prefix = prefix + ' | ' if prefix is not None else ''
        if epoch is not None:
            self.prefix += f"epoch {epoch:02d}"
        self.iterable = iterable if self.quiet else tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.iterable)

    def log(self, stats, verbose=False):
        if not self.quiet:
            self.iterable.set_postfix(self.format_stats(stats, verbose), refresh=True)

    def format_stats(self, stats, verbose=False):
        postfix = OrderedDict(stats)
        for key, value in postfix.items():
            if isinstance(value, Number):
                fmt = "{:.3f}" if value > 0.001 else "{:.1e}"
                postfix[key] = fmt.format(value)
            elif isinstance(value, AverageMeter) or isinstance(value, RunningAverageMeter):
                if verbose:
                    postfix[key] = f"{value.avg:.3f} ({value.val:.3f})"
                else:
                    postfix[key] = f"{value.avg:.3f}"
            elif isinstance(value, TimeMeter):
                postfix[key] = f"{value.elapsed_time:.1f}s"
            elif not isinstance(postfix[key], str):
                postfix[key] = str(value)
        return postfix

    def print(self, stats, verbose=False):
        postfix = " | ".join(key + " " + value.strip() for key, value in self.format_stats(stats, verbose).items())
        return f"{self.prefix + ' | ' if self.epoch is not None else ''}{postfix}"
            
    
