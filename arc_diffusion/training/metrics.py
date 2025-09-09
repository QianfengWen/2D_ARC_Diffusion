"""Evaluation metrics for ARC diffusion."""

import torch


COLOR_CLASSES = 10


@torch.no_grad()
def evaluate(model, diffusion, loader, device, max_batches=10):
    """Evaluate model on validation data.
    
    Args:
        model: The UNet model
        diffusion: The diffusion process
        loader: DataLoader for validation data
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        Tuple of (pixel_accuracy, problem_accuracy)
    """
    model.eval()
    pix_correct = 0
    pix_total = 0
    prob_correct = 0
    prob_total = 0
    seen = 0
    
    for ctx_in, ctx_out, q_in, q_out_oh, q_out_idx in loader:
        ctx_in = ctx_in.to(device)
        ctx_out = ctx_out.to(device)
        q_in = q_in.to(device)
        q_out_idx = q_out_idx.to(device)
        
        B, _, S, _ = q_in.shape
        
        # Sample from the model
        x0 = diffusion.sample(model, q_in, (B, COLOR_CLASSES, S, S), ctx_in, ctx_out)
        pred = x0.argmax(dim=1)  # (B,S,S)
        
        # Pixel accuracy
        pix_correct += (pred == q_out_idx).sum().item()
        pix_total += q_out_idx.numel()
        
        # Problem accuracy (all pixels must be correct)
        prob_correct += (pred.view(B,-1) == q_out_idx.view(B,-1)).all(dim=1).sum().item()
        prob_total += B
        
        seen += 1
        if seen >= max_batches:
            break
    
    pix_acc = pix_correct / max(1, pix_total)
    prob_acc = prob_correct / max(1, prob_total)
    
    return pix_acc, prob_acc
