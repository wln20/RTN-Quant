import torch

def quantize_tensor(w, n_bit, q_group_size):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape)
    scales = scales.reshape(org_w_shape[0], -1)
    zeros = zeros.reshape(org_w_shape[0], -1)
    return w, scales, zeros

def write_8bit_tensor(w_q):
    w_q = w_q.type(torch.uint8)
    return w_q

def write_4bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 2)
    w_int4 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    w_int4[:, 0] = w_q[:, 0] << 4
    w_int4[:, 0] += w_q[:, 1]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 4 // 8,)
    return w_int4.reshape(new_shape)

def write_3bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 8)
    w_int3 = torch.zeros(w_q.shape[0], 3, dtype=torch.uint8, device=w_q.device)

    # byte 1
    w_int3[:, 0] = w_q[:, 0] << 5
    w_int3[:, 0] += w_q[:, 1] << 2
    w_int3[:, 0] += w_q[:, 2] >> 1

    # byte 2
    w_int3[:, 1] = w_q[:, 2] << 7
    w_int3[:, 1] += w_q[:, 3] << 4
    w_int3[:, 1] += w_q[:, 4] << 1
    w_int3[:, 1] += w_q[:, 5] >> 2

    # byte 3
    w_int3[:, 2] = w_q[:, 5] << 6
    w_int3[:, 2] += w_q[:, 6] << 3
    w_int3[:, 2] += w_q[:, 7]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 3 // 8,)
    return w_int3.reshape(new_shape)

def write_2bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 4)
    w_int2 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    w_int2[:, 0] = w_q[:, 0] << 6
    w_int2[:, 0] += w_q[:, 1] << 4
    w_int2[:, 0] += w_q[:, 2] << 2
    w_int2[:, 0] += w_q[:, 3]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 2 // 8,)
    return w_int2.reshape(new_shape)

def write_1bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.type(torch.uint8).reshape(-1, 8)
    w_int1 = torch.zeros(w_q.shape[0], 1, dtype=torch.uint8, device=w_q.device)

    w_int1[:, 0] = w_q[:, 0] << 7
    w_int1[:, 0] += w_q[:, 1] << 6
    w_int1[:, 0] += w_q[:, 2] << 5
    w_int1[:, 0] += w_q[:, 3] << 4
    w_int1[:, 0] += w_q[:, 4] << 3
    w_int1[:, 0] += w_q[:, 5] << 2
    w_int1[:, 0] += w_q[:, 6] << 1
    w_int1[:, 0] += w_q[:, 7]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 1 // 8,)
    return w_int1.reshape(new_shape)

def deq_8bit_tensor(w_int8):
    return w_int8

def deq_4bit_tensor(w_int4):
    w_int4_org_shape = w_int4.shape
    new_shape = w_int4_org_shape[:-1] + (w_int4_org_shape[-1] * 8 // 4,)
    w_int4 = w_int4.reshape(-1, 1)
    w_q = torch.zeros(w_int4.shape[0], 2, dtype=torch.uint8, device=w_int4.device)

    w_q[:, 0] = w_int4[:, 0] >> 4
    w_q[:, 1] = (w_int4[:, 0] << 4) >> 4

    return w_q.reshape(new_shape)

def deq_3bit_tensor(w_int3):
    w_int3_org_shape = w_int3.shape
    new_shape = w_int3_org_shape[:-1] + (w_int3_org_shape[-1] * 8 // 3,)
    w_int3 = w_int3.reshape(-1, 3)
    w_q = torch.zeros(w_int3.shape[0], 8, dtype=torch.uint8, device=w_int3.device)

    w_q[:, 0] = w_int3[:, 0] >> 5
    w_q[:, 1] = (w_int3[:, 0] << 3) >> 5
    w_q[:, 2] = ((w_int3[:, 0] << 6) >> 5) + (w_int3[:, 1] >> 7)
    w_q[:, 3] = (w_int3[:, 1] << 1) >> 5
    w_q[:, 4] = (w_int3[:, 1] << 4) >> 5
    w_q[:, 5] = ((w_int3[:, 1] << 7) >> 5) + (w_int3[:, 2] >> 6)
    w_q[:, 6] = (w_int3[:, 2] << 2) >> 5
    w_q[:, 7] = (w_int3[:, 2] << 5) >> 5

    return w_q.reshape(new_shape)

def deq_2bit_tensor(w_int2):
    w_int2_org_shape = w_int2.shape
    new_shape = w_int2_org_shape[:-1] + (w_int2_org_shape[-1] * 8 // 2,)
    w_int2 = w_int2.reshape(-1, 1)
    w_q = torch.zeros(w_int2.shape[0], 4, dtype=torch.uint8, device=w_int2.device)

    w_q[:, 0] = w_int2[:, 0] >> 6
    w_q[:, 1] = (w_int2[:, 0] << 2) >> 6
    w_q[:, 2] = (w_int2[:, 0] << 4) >> 6
    w_q[:, 3] = (w_int2[:, 0] << 6) >> 6

    return w_q.reshape(new_shape)

def deq_1bit_tensor(w_int1):
    w_int1_org_shape = w_int1.shape
    new_shape = w_int1_org_shape[:-1] + (w_int1_org_shape[-1] * 8 // 1,)
    w_int1 = w_int1.reshape(-1, 1)
    w_q = torch.zeros(w_int1.shape[0], 8, dtype=torch.uint8, device=w_int1.device)

    w_q[:, 0] = w_int1[:, 0] >> 7
    w_q[:, 1] = (w_int1[:, 0] << 1) >> 7
    w_q[:, 2] = (w_int1[:, 0] << 2) >> 7
    w_q[:, 3] = (w_int1[:, 0] << 3) >> 7
    w_q[:, 4] = (w_int1[:, 0] << 4) >> 7
    w_q[:, 5] = (w_int1[:, 0] << 5) >> 7
    w_q[:, 6] = (w_int1[:, 0] << 6) >> 7
    w_q[:, 7] = (w_int1[:, 0] << 7) >> 7

    return w_q.reshape(new_shape)