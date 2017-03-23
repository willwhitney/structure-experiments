import torch
import socket

if socket.gethostname() == 'zaan':
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

batch_size = 32
