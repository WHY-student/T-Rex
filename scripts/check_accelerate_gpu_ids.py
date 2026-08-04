import os
import torch
from accelerate import Accelerator

acc = Accelerator()
idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
name = torch.cuda.get_device_name(idx) if idx >= 0 else "cpu"
print(f"rank={acc.process_index} local={acc.local_process_index} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} cuda={idx} name={name}")
