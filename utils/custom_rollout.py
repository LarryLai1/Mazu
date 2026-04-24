import dataclasses
import torch

def rollout_with_gpu(
        model,
        batch,
        steps,
    ):
    """Perform a roll-out to make long-term predictions.
    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.
    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    for _ in range(steps):
        pred = model(batch)
        yield pred
        batch = dataclasses.replace(
            pred,
            surf_vars = {
                k: torch.cat([batch.surf_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars = {
                k: torch.cat([batch.atmos_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.atmos_vars.items()
            },
        )

def rollout_with_gpu_toy(
        model,
        batch,
        steps,
        ground_truth_list=None,  # Pass _label_list here
    ):
    """
    Perform a roll-out. If ground_truth_list is provided, use it for 
    Teacher Forcing (updating history with actual data).
    """
    # 1. Init
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    # 2. Loop
    # Change '_' to 'i' so we can use the index to find the matching label
    for i in range(steps):
        pred = model(batch)
        yield pred
        
        # --- PREPARE INPUT FOR NEXT STEP ---
        
        if ground_truth_list is not None:
            # === TEACHER FORCING MODE ===
            
            # Step 1: Get the ground truth for the CURRENT step 'i'
            # (If i=0, we just predicted T+1. We pick the label for T+1 
            #  to serve as history for the next prediction.)
            gt_step = ground_truth_list[i] 

            batch = dataclasses.replace(
                pred, # Template (Metadata from pred)
                surf_vars={
                    # Access gt_step directly. It is already the dict for this timestamp.
                    k: torch.cat([batch.surf_vars[k][:, 1:], gt_step["surf_vars"][k]], dim=1)
                    for k in batch.surf_vars.keys()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], gt_step["atmos_vars"][k]], dim=1)
                    for k in batch.atmos_vars.keys()
                },
            )

        else:
            # === AUTOREGRESSIVE MODE (Original) ===
            # Use the model's own prediction 'pred'
            
            batch = dataclasses.replace(
                pred,
                surf_vars={
                    k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.surf_vars.items()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.atmos_vars.items()
                },
            )

def rollout_with_gpu_toy_noise(
        model,
        batch,
        steps,
        ground_truth_list=None,
        # --- New Arguments for Noise ---
        gaussian_noise_std=0.0, 
        scales=None,            # Dict mapping var_name -> std value
        levels=None             # List/Tensor of pressure levels [50, 100, ...]
    ):
    """
    Perform a roll-out. 
    If ground_truth_list is provided, uses Teacher Forcing.
    If gaussian_noise_std > 0, adds noise to the GT before feeding it back.
    """
    # 1. Init
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    # 2. Loop
    for i in range(steps):
        pred = model(batch)
        yield pred
        
        # --- PREPARE INPUT FOR NEXT STEP ---
        
        if ground_truth_list is not None:
            # === TEACHER FORCING MODE ===
            
            # Get the ground truth for the CURRENT step
            gt_step = ground_truth_list[i] 

            # --- [START NOISE INJECTION] ---
            # We must use a separate variable so we don't mutate ground_truth_list in-place
            if gaussian_noise_std > 0.0 and scales is not None:
                
                # Create a container for the noisy data
                gt_step_noisy = {
                    "surf_vars": {},
                    "atmos_vars": {}
                }

                # A. Surface Variables
                for var_name, tensor in gt_step["surf_vars"].items():
                    # tensor shape is likely [B, 1, H, W]
                    
                    if var_name in scales:
                        fixed_std = scales[var_name]
                        
                        # Generate Noise
                        noise = torch.randn_like(tensor) * fixed_std * gaussian_noise_std
                        
                        # Add to a CLONE of the data (to avoid corrupting original dataset)
                        gt_step_noisy["surf_vars"][var_name] = tensor.clone() + noise
                    else:
                        gt_step_noisy["surf_vars"][var_name] = tensor

                # B. Atmospheric Variables
                # tensor shape is likely [B, 1, Levels, H, W]
                for var_name, tensor in gt_step["atmos_vars"].items():
                    
                    # Build scale tensor for the levels
                    level_scales = []
                    found_levels = True
                    
                    # Ensure we have the levels list to map keys like "t_50"
                    current_levels = levels if levels is not None else []
                    
                    for lvl in current_levels:
                        key = f"{var_name}_{int(lvl)}" # e.g. "t_50"
                        if key in scales:
                            level_scales.append(scales[key])
                        else:
                            found_levels = False
                            break
                    
                    if found_levels:
                        # Convert list to tensor: [L] -> [1, 1, L, 1, 1]
                        scale_tensor = torch.tensor(level_scales, device=tensor.device, dtype=tensor.dtype)
                        scale_tensor = scale_tensor.view(1, 1, -1, 1, 1)

                        # Generate Noise
                        # Note: 'tensor' here is the GT slice.
                        noise = torch.randn_like(tensor) * scale_tensor * gaussian_noise_std
                        
                        gt_step_noisy["atmos_vars"][var_name] = tensor.clone() + noise
                    else:
                        gt_step_noisy["atmos_vars"][var_name] = tensor

                # Use the noisy version for update
                current_step_data = gt_step_noisy

            else:
                # No noise, use pure Ground Truth
                current_step_data = gt_step
            # --- [END NOISE INJECTION] ---

            # Update History
            batch = dataclasses.replace(
                pred,
                surf_vars={
                    k: torch.cat([batch.surf_vars[k][:, 1:], current_step_data["surf_vars"][k]], dim=1)
                    for k in batch.surf_vars.keys()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], current_step_data["atmos_vars"][k]], dim=1)
                    for k in batch.atmos_vars.keys()
                },
            )

        else:
            # === AUTOREGRESSIVE MODE ===
            batch = dataclasses.replace(
                pred,
                surf_vars={
                    k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.surf_vars.items()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.atmos_vars.items()
                },
            )



def rollout_with_multiple_gpu(
        model,
        unwrap_model,
        batch,
        steps,
    ):
    """Perform a roll-out to make long-term predictions.
    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        unwrap_model (:class:`aurora.model.aurora.Aurora`): The unwrapped model to access attributes.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.
    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    batch = unwrap_model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(unwrap_model.patch_size)
    batch = batch.to(p.device)

    for _ in range(steps):
        pred = model(batch)
        yield pred
        batch = dataclasses.replace(
            pred,
            surf_vars = {
                k: torch.cat([batch.surf_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars = {
                k: torch.cat([batch.atmos_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.atmos_vars.items()
            },
        )