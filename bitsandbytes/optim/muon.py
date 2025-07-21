from typing import Optional
from bitsandbytes.optim.optimizer import Optimizer1State


class Muon(Optimizer1State):
    def __init__(
        self,
        params,
        lr,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        optim_bits: int = 32,
        args: Optional[object] = None,
        min_8bit_size: int = 4096,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        ns_steps: int = 1,
    ):
        """
        Base Muon optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`):
                The learning rate.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            dampening (`float`, defaults to 0):
                The dampening value reduces the momentum of the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            nesterov (`bool`, defaults to `False`):
                Whether to use Nesterov momentum.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            ns_steps (`int`, defaults to 1):
                The number of Newton-Schulz steps to use.
        """
        if momentum == 0:
            raise NotImplementedError("SGD without momentum is not supported!")
        super().__init__(
            "muon",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )


# class SGD8bit(Optimizer1State):
#     def __init__(
#         self,
#         params,
#         lr,
#         momentum=0,
#         dampening=0,
#         weight_decay=0,
#         nesterov=False,
#         args=None,
#         min_8bit_size=4096,
#         percentile_clipping=100,
#         block_wise=True,
#     ):
#         """
#         8-bit SGD optimizer.

#         Arguments:
#             params (`torch.tensor`):
#                 The input parameters to optimize.
#             lr (`float`):
#                 The learning rate.
#             momentum (`float`, defaults to 0):
#                 The momentum value speeds up the optimizer by taking bigger steps.
#             dampening (`float`, defaults to 0):
#                 The dampening value reduces the momentum of the optimizer.
#             weight_decay (`float`, defaults to 0.0):
#                 The weight decay value for the optimizer.
#             nesterov (`bool`, defaults to `False`):
#                 Whether to use Nesterov momentum.
#             args (`object`, defaults to `None`):
#                 An object with additional arguments.
#             min_8bit_size (`int`, defaults to 4096):
#                 The minimum number of elements of the parameter tensors for 8-bit optimization.
#             percentile_clipping (`int`, defaults to 100):
#                 Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
#             block_wise (`bool`, defaults to `True`):
#                 Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
#         """
#         if momentum == 0:
#             raise NotImplementedError("SGD without momentum is not supported!")
#         super().__init__(
#             "momentum",
#             params,
#             lr,
#             (momentum, dampening),
#             0.0,
#             weight_decay,
#             8,
#             args,
#             min_8bit_size,
#             percentile_clipping,
#             block_wise,
#         )


class Muon32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        optim_bits: int = 32,
        args: Optional[object] = None,
        min_8bit_size: int = 4096,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        ns_steps: int = 1,
    ):
        """
        Base Muon optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`):
                The learning rate.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            dampening (`float`, defaults to 0):
                The dampening value reduces the momentum of the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            nesterov (`bool`, defaults to `False`):
                Whether to use Nesterov momentum.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            ns_steps (`int`, defaults to 1):
                The number of Newton-Schulz steps to use.
        """
        if momentum == 0:
            raise NotImplementedError("SGD without momentum is not supported!")
        super().__init__(
            "muon",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )