import numpy as np
import torch


def cal_amplitude(real, imag, model=np):
    return model.sqrt(real*real+imag*imag)


def cal_phase(real, imag, model=np):
    return model.arctan2(real, imag)


def sequence_to_vector(img: np.ndarray | torch.Tensor, is_db=False):
    assert len(img.shape) == 3
    c = img.shape[0]
    assert c == 9
    if isinstance(img, np.ndarray):
        model = np
    elif isinstance(img, torch.Tensor):
        model = torch
    data = model.zeros_like(img)
    eps=1e-15
    if is_db:
        data[0:3, :, :] = 10*model.log10(img[0:3, :, :]+eps)
    else:
        data[0:3, :, :] = img[0:3, :, :]
    for i in range(3, 9, 2):
        if is_db:
            data[i, :, :] = 10*model.log10(cal_amplitude(img[i], img[i+1], model)+eps)
        else:
            data[i, :, :] = cal_amplitude(img[i], img[i+1], model)
        data[i+1, :, :] = cal_phase(img[i+1], img[i], model)
    # for i in range(3, 6):
    #     data[i, :, :] = cal_amplitude(img[i], img[i+3], model)
    #     data[i+3, :, :] = cal_phase(img[i], img[i+3], model)
    return data


def sequence_to_complex(img: np.ndarray | torch.Tensor):
    assert len(img.shape) == 3
    c, h, w = img.shape
    assert c == 9
    if isinstance(img, np.ndarray):
        data = np.zeros((6, h, w), dtype=complex)
    elif isinstance(img, torch.Tensor):
        data = torch.zeros((6, h, w), dtype=torch.float) + \
            torch.zeros((6, h, w), dtype=torch.float)*1j
    data[0:3, :, :] = img[0:3, :, :]+0j
    for i in range(3, 9, 2):
        data[i//2+2, :, :] = img[i, :, :]+img[i+1, :, :]*1j
    return data


def sequence_to_matrix(img: np.ndarray|torch.Tensor):
    assert len(img.shape) == 3
    c, h, w = img.shape
    assert c == 9    
    if isinstance(img, np.ndarray):
        data = np.zeros((3, 3, h, w), dtype=np.complex64)
        model = np
    elif isinstance(img, torch.Tensor):
        data = torch.zeros((3, 3, h, w), dtype=torch.float, device=img.device) + \
            torch.zeros((3, 3, h, w), dtype=torch.float, device=img.device)*1j
        model = torch
    for i in range(0, 3):
        data[i, i, :, :] = img[i, :, :]+0j
        for j in range(i+1, 3):
            data[i, j, :, :] = img[(i+j)*2+1, :, :]+img[(i+j)*2+2, :, :]*1j
            data[j, i, :, :] = model.conj(data[i, j, :, :])
    return data

def matrix_to_sequence(img: np.ndarray|torch.Tensor):
    assert len(img.shape) == 4
    _, _, h, w = img.shape
    if isinstance(img, np.ndarray):
        data = np.zeros((9, h, w), dtype=float) 
    elif isinstance(img, torch.Tensor):
        data = torch.zeros((9, h, w), dtype=torch.float, device=img.device)

    for i in range(0, 3):
        data[i, :, :] = img[i, i, :, :].real
        for j in range(i+1, 3):
            data[(i+j)*2+1, :, :] = img[i, j, :, :].real
            data[(i+j)*2+2, :, :] = img[i, j, :, :].imag
    return data