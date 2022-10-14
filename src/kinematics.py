from typing import Tuple
from time import time
import numpy as np

import thirdparty.FrEIA.framework as Ff
import numpy as np
import torch

PI = np.pi


def forward_kinematics(x: torch.Tensor) -> torch.Tensor:
    arm_length = 1.0
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    batch_sz = x.shape[0]

    fk_x = (
        arm_length * torch.cos(x1)
        + arm_length * torch.cos(x1 + x2)
        + arm_length * torch.cos(x1 + x2 + x3)
        + arm_length * torch.cos(x1 + x2 + x3 + x4)
    )
    fk_x = fk_x.reshape((batch_sz, 1))

    fk_y = (
        arm_length * torch.sin(x1)
        + arm_length * torch.sin(x1 + x2)
        + arm_length * torch.sin(x1 + x2 + x3)
        + arm_length * torch.sin(x1 + x2 + x3 + x4)
    )
    fk_y = fk_y.reshape((batch_sz, 1))

    return torch.cat([fk_x, fk_y], dim=1)


def inverse_kinematics(
    model: Ff.ReversibleGraphNet, y: torch.Tensor, ndofs: int, dim_z: int, device="cuda"
) -> Tuple[torch.Tensor, float]:
    """IK wahoo!!
    Returns:
        Tuple[torch.Tensor, float]: solutions and runtime
    """
    t0 = time()
    batch_size = y.shape[0]
    conditional = y
    latent = torch.randn((batch_size, dim_z)).to(device)
    output_rev, _ = model(latent, c=conditional, rev=True)
    return output_rev[:, 0:ndofs], time() - t0


def run_tests(model: Ff.ReversibleGraphNet, dim_z: int):
    # All 0
    x = torch.zeros((5, dim_z))
    y_expected = torch.tensor([[4, 0], [4, 0], [4, 0], [4, 0], [4, 0]])
    y_calculated = forward_kinematics(x)
    torch.testing.assert_allclose(y_expected, y_calculated)

    # First angle is pi/2, rest are 0
    x = torch.zeros((5, dim_z))
    x[:, 0] = np.pi / 2
    y_expected = torch.tensor([[0, 4], [0, 4], [0, 4], [0, 4], [0, 4]])
    y_calculated = forward_kinematics(x)
    torch.testing.assert_allclose(y_expected, y_calculated)
    print("Forward kinematics unit tests passed")


def calculate_ave_l2_errror(model: Ff.ReversibleGraphNet, ndofs: int, dim_z: int, device="cuda") -> Tuple[float, float]:
    N = 100
    x_ground_truth = 2 * PI * torch.rand((N, ndofs), device=device) - PI
    y_ground_truth = forward_kinematics(x_ground_truth).to(device)
    x_calculated, runtime = inverse_kinematics(model, y_ground_truth, ndofs, dim_z)
    y_realized = forward_kinematics(x_calculated)
    positional_errors = np.linalg.norm(
        y_ground_truth.detach().cpu().numpy() - y_realized.detach().cpu().numpy(), axis=1
    )
    return np.mean(positional_errors), runtime
