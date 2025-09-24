"""Evaluation metrics for ARC diffusion."""

import torch


COLOR_CLASSES = 10


@torch.no_grad()
def evaluate(model, diffusion, loader, device, max_batches=10):
    """Evaluate model on validation data with mode-specific metrics.
    
    Args:
        model: The UNet model
        diffusion: The diffusion process
        loader: DataLoader for validation data
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        Tuple of (pixel_accuracy, problem_accuracy, avg_val_loss, per_task)
        where `per_task` is either None (no task ids available) or a dict
        mapping int task_id -> {"pix_acc": float, "prob_acc": float}.
        
    Note:
        - Baseline mode: standard pixel/problem accuracy 
        - Occupancy mode: position accuracy (occupied/unoccupied correct) and occupancy sample accuracy
        - Color mode: color accuracy only on occupied pixels
    """
    model.eval()
    pix_correct = 0
    pix_total = 0
    prob_correct = 0
    prob_total = 0
    loss_sum = 0.0
    loss_batches = 0
    seen = 0

    # Track per-task metrics if the loader provides task ids
    per_task_counts = {}  # tid -> {pc, pt, bc, bt}
    
    for batch in loader:
        # Support loaders with or without task ids
        if len(batch) == 6:
            ctx_in, ctx_out, q_in, q_out_oh, q_out_idx, q_tid = batch
        else:
            ctx_in, ctx_out, q_in, q_out_oh, q_out_idx = batch
            q_tid = None

        ctx_in = ctx_in.to(device)
        ctx_out = ctx_out.to(device)
        q_in = q_in.to(device)
        q_out_oh = q_out_oh.to(device)
        q_out_idx = q_out_idx.to(device)
        
        B, _, S, _ = q_in.shape
        
        # Sample from the model for accuracy metrics
        x0 = diffusion.sample(model, q_in, (B, COLOR_CLASSES, S, S), ctx_in, ctx_out)
        pred = x0.argmax(dim=1)  # (B,S,S)
        
        # Mode-specific accuracy calculation
        if diffusion.mode == "occupancy":
            # Occupancy mode: position accuracy (occupied/unoccupied correct)
            target_occupied = (q_out_idx > 0)  # (B,S,S) - True if occupied (non-black)
            pred_occupied = (pred > 0)         # (B,S,S) - True if predicted occupied
            
            # Position accuracy: correctly predicted occupied/unoccupied positions
            position_correct = (target_occupied == pred_occupied).sum().item()
            pix_correct += position_correct
            pix_total += q_out_idx.numel()
            
            # Sample accuracy: all positions correctly classified as occupied/unoccupied
            batch_position_correct = (target_occupied.view(B, -1) == pred_occupied.view(B, -1)).all(dim=1)
            prob_correct += batch_position_correct.sum().item()
            prob_total += B
            
        elif diffusion.mode == "color":
            # Color mode: accuracy only on occupied pixels (ignore unoccupied)
            occupied_mask = (q_out_idx > 0)  # (B,S,S) - mask for occupied pixels
            
            if occupied_mask.any():
                # Pixel accuracy: correct colors at occupied positions only
                occupied_correct = ((pred == q_out_idx) & occupied_mask).sum().item()
                occupied_total = occupied_mask.sum().item()
                pix_correct += occupied_correct
                pix_total += occupied_total
                
                # Sample accuracy: all occupied pixels correctly colored
                for b in range(B):
                    b_occupied_mask = occupied_mask[b]  # (S,S)
                    if b_occupied_mask.any():
                        b_correct = (pred[b][b_occupied_mask] == q_out_idx[b][b_occupied_mask]).all()
                        if b_correct:
                            prob_correct += 1
                    else:
                        # No occupied pixels - consider as correct
                        prob_correct += 1
                prob_total += B
            else:
                # No occupied pixels in entire batch - consider all correct
                pix_correct += 0  # No pixels to evaluate
                pix_total += 0
                prob_correct += B
                prob_total += B
                
        else:
            # Baseline mode: standard pixel/problem accuracy
            pix_correct += (pred == q_out_idx).sum().item()
            pix_total += q_out_idx.numel()
            
            # Problem accuracy (all pixels must be correct)
            batch_all_correct = (pred.view(B, -1) == q_out_idx.view(B, -1)).all(dim=1)  # (B,)
            prob_correct += batch_all_correct.sum().item()
            prob_total += B
        
        # Validation loss (same objective as training)
        loss = diffusion.compute_loss(model, (ctx_in, ctx_out, q_in, q_out_oh, q_out_idx))
        loss_sum += float(loss.item())
        loss_batches += 1

        # Per-task breakdown if task ids are available (mode-specific)
        if q_tid is not None:
            try:
                tid_vec = q_tid.detach().cpu().to(torch.int64)
            except Exception:
                tid_vec = torch.as_tensor(q_tid, dtype=torch.int64)
            
            # Mode-specific per-sample metrics
            if diffusion.mode == "occupancy":
                # Position accuracy per sample
                target_occupied = (q_out_idx > 0)
                pred_occupied = (pred > 0)
                per_sample_pix = (target_occupied.view(B, -1) == pred_occupied.view(B, -1)).sum(dim=1).detach().cpu().to(torch.int64)
                per_sample_tot = torch.full_like(per_sample_pix, S * S)
                per_sample_prob = batch_position_correct.detach().cpu().to(torch.int64)
                
            elif diffusion.mode == "color":
                # Color accuracy only on occupied pixels per sample
                per_sample_pix = []
                per_sample_tot = []
                per_sample_prob = []
                
                for b in range(B):
                    b_occupied_mask = (q_out_idx[b] > 0)
                    if b_occupied_mask.any():
                        b_correct_pixels = ((pred[b] == q_out_idx[b]) & b_occupied_mask).sum().item()
                        b_total_pixels = b_occupied_mask.sum().item()
                        b_all_correct = (pred[b][b_occupied_mask] == q_out_idx[b][b_occupied_mask]).all().item()
                    else:
                        # No occupied pixels
                        b_correct_pixels = 0
                        b_total_pixels = 0
                        b_all_correct = 1  # Consider as correct
                    
                    per_sample_pix.append(b_correct_pixels)
                    per_sample_tot.append(b_total_pixels)
                    per_sample_prob.append(b_all_correct)
                
                per_sample_pix = torch.tensor(per_sample_pix, dtype=torch.int64)
                per_sample_tot = torch.tensor(per_sample_tot, dtype=torch.int64)
                per_sample_prob = torch.tensor(per_sample_prob, dtype=torch.int64)
                
            else:
                # Baseline mode: standard metrics
                per_sample_pix = (pred == q_out_idx).view(B, -1).sum(dim=1).detach().cpu().to(torch.int64)
                per_sample_tot = torch.full_like(per_sample_pix, S * S)
                per_sample_prob = batch_all_correct.detach().cpu().to(torch.int64)
            
            ones = torch.ones_like(per_sample_prob)

            for i in range(B):
                tid = int(tid_vec[i].item())
                d = per_task_counts.setdefault(tid, {"pc": 0, "pt": 0, "bc": 0, "bt": 0})
                d["pc"] += int(per_sample_pix[i].item())
                d["pt"] += int(per_sample_tot[i].item())
                d["bc"] += int(per_sample_prob[i].item())
                d["bt"] += int(ones[i].item())
        
        seen += 1
        if seen >= max_batches:
            break
    
    pix_acc = pix_correct / max(1, pix_total)
    prob_acc = prob_correct / max(1, prob_total)
    avg_loss = loss_sum / max(1, loss_batches)

    per_task = None
    if per_task_counts:
        per_task = {}
        for tid, counts in per_task_counts.items():
            pc = counts["pc"]
            pt = counts["pt"]
            bc = counts["bc"]
            bt = counts["bt"]
            per_task[tid] = {
                "pix_acc": (pc / max(1, pt)),
                "prob_acc": (bc / max(1, bt)),
            }
    
    return pix_acc, prob_acc, avg_loss, per_task
