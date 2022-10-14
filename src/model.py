from typing import Dict

import thirdparty.FrEIA.framework as Ff
import thirdparty.FrEIA.modules as Fm
from thirdparty.FrEIA.framework.graph_inn import GraphINN
import torch
import torch.nn as nn


def glow_subnet(internal_size: int, n_layers: int, ch_in: int, ch_out: int):
    """Create a coefficient network"""

    assert n_layers in [1, 2, 3, 4, 5, 6, 7], "Number of layers `n_layers` must be in [1, ..., 7]"

    if n_layers == 1:
        return nn.Sequential(nn.Linear(ch_in, internal_size), nn.LeakyReLU(), nn.Linear(internal_size, ch_out))

    if n_layers == 2:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 3:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 4:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 5:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 6:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 7:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )


def glow_cinn_model(hparams: Dict, n_dofs: int, device="cuda") -> Ff.ReversibleGraphNet:
    """Build a nn_model consisting of a sequence of Glow coupling layers

    Args:
        hparams (Dict): Hyper-parameters of the model. Expected format:
            {
                "nb_nodes": int,
                "dim_latent_space": int,
                "coeff_fn_config": int,
                "coeff_fn_internal_size": int,
            }
    """

    def glow_subnet_wrapper(ch_in: int, ch_out: int):
        return glow_subnet(hparams["coeff_fn_internal_size"], hparams["coeff_fn_config"], ch_in, ch_out)

    # Input Node
    ndim_tot = hparams["dim_latent_space"]
    dim_cond = 2  # (x,y)
    input_node = Ff.InputNode(ndim_tot, name="input")
    nodes = [input_node]
    cond = Ff.ConditionNode(dim_cond)

    # Transform Node to map x_i from joint space to [-1, 1]
    x_invSig = torch.eye(ndim_tot)
    x_Mu = torch.zeros(ndim_tot)
    for i in range(n_dofs):
        x_invSig[i, i] = 1.0 / 3.141592
    nodes.append(Ff.Node([nodes[-1].out0], Fm.FixedLinearTransform, {"M": x_invSig, "b": x_Mu}))

    for i in range(hparams["nb_nodes"]):
        permute_node = Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {"seed": i})
        nodes.append(permute_node)

        glow_node = Ff.Node(
            nodes[-1].out0,
            Fm.GLOWCouplingBlock,
            {"subnet_constructor": glow_subnet_wrapper, "clamp": 2.5},
            conditions=cond,
        )

        nodes.append(glow_node)

    model = Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode([nodes[-1].out0], name="output")], verbose=False)
    model.to(device)
    return model
