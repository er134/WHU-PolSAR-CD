import math
from einops import rearrange
import torch
import numpy as np


def t3_to_c3(T3):
    # Define the transformation matrix U for Pauli to linear basis conversion
    # $$\sqrt{2}U_{3(L\rightarrow P)} = \begin{pmatrix}1&0&1\\1&0&-1\\0&\sqrt{2}&0\end{pmatrix}$$
    sqrt_2_U = np.array([[1, 0, 1],
                         [1, 0, -1],
                         [0, np.sqrt(2), 0]])
    
    _, _, h, w = T3.shape
    if isinstance(T3, np.ndarray):
        sqrt_2_U = np.tile(sqrt_2_U, (h, w, 1, 1))
        sqrt_2_U_T = sqrt_2_U.conj().transpose((0, 1, 3, 2))
        module = np
    elif isinstance(T3, torch.Tensor):
        sqrt_2_U = torch.from_numpy(sqrt_2_U).to(T3.device)
        sqrt_2_U = torch.tile(sqrt_2_U, (h, w, 1, 1))
        sqrt_2_U_T = sqrt_2_U.conj().transpose(sqrt_2_U, 2, 3)
        module = torch

    # Calculate C3 = U * T3 * U^H
    T3 = rearrange(T3, 'mh mw h w -> h w mh mw')
    temp = module.matmul(sqrt_2_U_T, T3) 
    temp = module.matmul(temp, sqrt_2_U)
    C3 = temp / 2
    C3 = rearrange(C3, 'h w mh mw -> mh mw h w')
    return C3

def c3toc2(C3, mode=(1,2)):
    x, y = mode
    index = (x, y, (x+y)*2+1, (x+y)*2+2)
    C2 = C3[index,:,:]
    return C2

