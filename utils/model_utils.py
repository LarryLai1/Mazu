import torch

class Muon(torch.optim.Optimizer):
    """A lightweight Muon-style optimizer for PyTorch versions without built-in Muon.

    This implementation applies momentum updates and projects matrix-like updates
    to an orthonormal basis via QR decomposition.
    """

    def __init__(
        self,
        params,
        lr = 1e-3,
        momentum = 0.95,
        weight_decay = 0.0,
        nesterov = True,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr = lr,
            momentum = momentum,
            weight_decay = weight_decay,
            nesterov = nesterov,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(update: torch.Tensor) -> torch.Tensor:
        original_shape = update.shape
        if update.ndim < 2:
            return update

        matrix = update.reshape(update.shape[0], -1)
        matrix_fp32 = matrix.float()

        if matrix_fp32.shape[0] >= matrix_fp32.shape[1]:
            q, _ = torch.linalg.qr(matrix_fp32, mode = "reduced")
            ortho = q
        else:
            q, _ = torch.linalg.qr(matrix_fp32.t(), mode = "reduced")
            ortho = q.t()

        return ortho.reshape(original_shape).to(dtype = update.dtype)

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(grad).detach()
                else:
                    state["momentum_buffer"].mul_(momentum).add_(grad)

                update = state["momentum_buffer"]
                if nesterov:
                    update = grad.add(update, alpha = momentum)

                if param.ndim >= 2:
                    update = self._orthogonalize(update)

                param.add_(update, alpha = -lr)

        return loss