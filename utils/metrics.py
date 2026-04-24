import torch
from aurora.batch import Batch
from functools import partial
from einops import reduce, rearrange
from dataclasses import dataclass

def AuroraLoss(
    pred: Batch, 
    target: Batch,
    loss_function_type: str,
) -> dict:

    if loss_function_type in ("MAE", "L1Loss"):
        loss_function = torch.nn.L1Loss(reduction = "none")
    elif loss_function_type in ("MSE", "MSELoss"):
        loss_function = torch.nn.MSELoss(reduction = "none")
    else:
        raise ValueError(f"Unsupported loss function type: {loss_function_type}")

    loss_dict = {}

    all_per_sample_losses = []

    loss_dict["surf_vars"] = {}
    for k in pred.surf_vars:
        if k not in target.surf_vars:
            raise KeyError(f"{k} missing in target batch surf_vars.")
        loss = loss_function(pred.surf_vars[k], target.surf_vars[k])
        loss_per_sample = reduce(loss, "b t h w -> b", "mean")
        loss_dict["surf_vars"][k] = loss_per_sample.detach()
        all_per_sample_losses.append(loss_per_sample)

    loss_dict["atmos_vars"] = {}
    for k in pred.atmos_vars:
        if k not in target.atmos_vars:
            raise KeyError(f"{k} missing in target batch atmos_vars.")
        loss_dict["atmos_vars"][k] = {}
        for i, l in enumerate(pred.metadata.atmos_levels):
            loss = loss_function(pred.atmos_vars[k][:, :, i], target.atmos_vars[k][:, :, i])
            loss_per_sample = reduce(loss, "b t h w -> b", "mean")
            loss_dict["atmos_vars"][k][l] = loss_per_sample.detach()
            all_per_sample_losses.append(loss_per_sample)

    if all_per_sample_losses:
        all_vars_tensor = rearrange(all_per_sample_losses, 'v b -> b v')
        all_vars_mean = reduce(all_vars_tensor, 'b v -> b', 'mean')
        loss_dict["all_vars"] = all_vars_mean
    else:
        loss_dict["all_vars"] = None

    return loss_dict

AuroraMAELoss = partial(AuroraLoss, loss_function_type = "MAE")
AuroraMSELoss = partial(AuroraLoss, loss_function_type = "MSE")

@dataclass
class MSEAggregator:
    error_sum: float = 0.0
    count: int = 0

    def update(self, error_value_tensor: torch.Tensor):
        error_value_tensor = error_value_tensor.detach()
        self.error_sum += error_value_tensor.sum().item()
        self.count += error_value_tensor.numel()

    def mean(self):
        if self.count == 0:
            return float("NaN")
        return self.error_sum / self.count

@dataclass
class MAEAggregator:
    error_sum: float = 0.0
    count: int = 0

    def update(self, error_value_tensor: torch.Tensor):
        error_value_tensor = error_value_tensor.detach()
        self.error_sum += error_value_tensor.sum().item()
        self.count += error_value_tensor.numel()

    def mean(self):
        if self.count == 0:
            return float("NaN")
        return self.error_sum / self.count

def prepare_each_lead_time_agg(
    rollout_step: int,
    lead_time: int,
    surface_variables: list,
    upper_variables: list,
    levels: list,
    err_type: str,
) -> dict:
    agg = {}
    var_name_mapping = {
        "t2m": "2t",
        "u10": "10u",
        "v10": "10v",
        "msl": "msl",
    }

    if err_type == "MSE":
        aggregator = MSEAggregator
    elif err_type == "MAE":
        aggregator = MAEAggregator

    # max_lead_time = rollout_step * lead_time
    # for t in range(1, max_lead_time + 1):
    for rs in range(1, rollout_step + 1):
        t = rs * lead_time
        agg[t] = {'surf_vars': {}, 'atmos_vars': {}}
        for var in surface_variables:
            _var = var_name_mapping[var] if var in var_name_mapping else var
            agg[t]['surf_vars'][_var] = aggregator(
                error_sum = 0.0,
                count = 0,
            )
        for var in upper_variables:
            _var = var_name_mapping[var] if var in var_name_mapping else var
            agg[t]['atmos_vars'][_var] = {}
            for lev in levels:
                agg[t]['atmos_vars'][_var][lev] = aggregator(
                    error_sum = 0.0,
                    count = 0,
                )
    return agg
