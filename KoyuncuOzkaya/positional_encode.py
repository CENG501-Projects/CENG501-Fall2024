import torch

def positional_encoding_sin_cos(p, num_freqs=10, include_input=True):
    """
    p: (..., 3)
    Returns a sinusoidal embedding of shape (..., 3 + 2 * 3 * num_freqs) 
    if include_input=True. 
    """
    # Example code for a standard NeRF-like encoding
    shape_p = p.shape
    if p.ndim == 1:
        p = p.unsqueeze(0)

    freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs, device=p.device)
    p = p.unsqueeze(-2)  # => [..., 1, 3]
    p_freq = p * freq_bands.view(1, -1, 1)  # => [..., num_freqs, 3]

    sin_p = p_freq.sin()
    cos_p = p_freq.cos()
    pe = torch.cat([sin_p, cos_p], dim=-1)  # => [..., num_freqs, 6]
    pe = pe.view(*pe.shape[:-2], -1)       # => [..., num_freqs*6]

    out_list = []
    if include_input:
        out_list.append(p.squeeze(-2))  # raw coords => [..., 3]
    out_list.append(pe)

    out = torch.cat(out_list, dim=-1)
    return out
