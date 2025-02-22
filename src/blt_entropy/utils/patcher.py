import torch
import torch.nn as nn
import torch.nn.functional as F


from contextlib import nullcontext

def entropy(scores):
    """_summary_
    This functions takes in logits and then returns the entropy
    Args:
        scores (FloatTensor): The logits
    Returns:
        FloatTensor: The entropy
        
    """
    log_probs = torch.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy


def calculate_entropies(
    tokens: torch.Tensor,
    entropy_model,
    patching_batch_size,
    device: str | None = None,
    enable_grad: bool = False,
):
    
    grad_context = nullcontext() if enable_grad else torch.no_grad()
    
    with grad_context:
        entropies = []
        preds = []
        max_length = getattr(entropy_model, "max_length", 8192)
        batch_numel = max_length * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)
        
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat([split, pad], dim=0)
            split = split.reshape(
                -1, max_length
            )
            if device is not None:
                split = split.to(device)
            
            assert torch.all(split >= 0) and torch.all(split < 260)
            pred = entropy_model(split)
            pred = pred.reshape(-1, pred.shape[-1])[
                : split.numel() - pad_size, :
            ] # [batch_size * seq_len, vocab]
            
            preds.append(pred)
            pred_entropy = entropy(pred)
            entropies.append(pred_entropy)
            
        concat_entropies = torch.cat(entropies, dim=0)
        concat_entropies = concat_entropies.reshape(tokens.shape)
        concat_preds = torch.cat(preds, dim=0)
        concat_preds = concat_preds.reshape(tokens.shape[0], tokens.shape[1], -1)
    return concat_entropies, concat_preds

def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """_summary_
    This function patches the start mask from the entropies
    Args:
        entropies (FloatTensor): The entropies
        t (float): The threshold
    Returns:
        ByteTensor: The mask
    """
    bs, seqlen = entropies.shape
    
    if seqlen == 0:
        return entropies > t
    
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True
    
    #calculate the differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]
    condition = differences > t
    mask[:,1:] = condition
    return mask

def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    bs, seqlen = entropies.shape
    
    if seqlen == 0:
        return entropies > t
    
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True
    
    differences = entropies[:, 1:] - entropies[:, :-1]
    condition = (differences > t_add) & (entropies[:,1:] > t) & (~mask[:,:-1])
    mask[:, 1:] = condition
    return mask


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seqlen = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            size=(bs, trunc_seqlen),
            fill_value=trunc_seqlen,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(
                trunc_seqlen, device=patch_start_mask.device
            )
            .unsqueeze(0)
            .repeat(bs, 1)
            
        )
        
        extra_patch_ids = torch.full(
            size = (bs, trunc_seqlen),
            fill_value=trunc_seqlen,
            dtype=torch.long,
            device=patch_start_mask.device
        )
        
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seqlen
        )[:, :max_patches]
        
        return patch_start_ids

def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True
):
    bs, seqlen = entropies.shape
    
    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    
    preds_truncation_len = first_ids.shape[1]
    entropies = entropies[:,1:]
    
    if threshold is None:
        num_patches = seqlen // patch_size
        patch_start_ids = entropies.topk(num_patches-2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
        
    else:
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(
                entropies=entropies, t=threshold
            )
        elif threshold_add is not None and threshold is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(
                entropies=entropies, t=threshold, t_add=threshold_add
            )
        
        else:
            patch_start_mask = entropies > threshold
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
    patch_start_ids = torch.cat(
        (first_ids, patch_start_ids+preds_truncation_len), dim=1
    )
    return patch_start_ids


class Patcher:
    def __init__(self):
        pass
    def patch(self, 
              tokens: torch.Tensor, 
              include_next_token: bool = False,
              preds: torch.Tensor | None = None,
              entropies: torch.Tensor | None = None) -> torch.Tensor:
        pass