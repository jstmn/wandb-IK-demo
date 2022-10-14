import argparse

from src.kinematics import calculate_ave_l2_errror, forward_kinematics, inverse_kinematics, run_tests, PI
from src.model import glow_cinn_model
import thirdparty.FrEIA.framework as Ff

import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt
from PIL import Image

NDOFS = 4
PLOT_TARGET_POSE = [2.5, 0]

LOG_LOSS_EVERY_K = 500
EVAL_EVERY_K = 1000
LOG_PLOT_EVERY_K = 1000
STEP_LR_EVERY = int(2.5 * 1e6 / 64)
GAMMA = 0.985
device = "cuda"


def plot_solutions(model: Ff.ReversibleGraphNet, ndofs: int, dim_z: int, global_step: int) -> Image:
    def draw_robot_on(ax, x, color, alpha=0.35):
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        p0 = np.array([0, 0])
        p1 = np.array([np.cos(x0), np.sin(x0)])
        p2 = p1 + np.array([np.cos(x0 + x1), np.sin(x0 + x1)])
        p3 = p2 + np.array([np.cos(x0 + x1 + x2), np.sin(x0 + x1 + x2)])
        p4 = p3 + np.array([np.cos(x0 + x1 + x2 + x3), np.sin(x0 + x1 + x2 + x3)])
        merged = np.vstack([p0, p1, p2, p3, p4])
        ax.plot(merged[:, 0], merged[:, 1], color=color, alpha=alpha)
        ax.scatter(merged[:-1, 0], merged[:-1, 1], color=color, alpha=alpha, s=5)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_xlabel(f"X")
    ax.set_ylabel(f"Y")
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    ax.scatter([PLOT_TARGET_POSE[0]], [PLOT_TARGET_POSE[1]], marker="o", color="g", s=50, alpha=1)

    y_batch = torch.zeros((50, 2), device=device)
    y_batch[:, 0] = PLOT_TARGET_POSE[0]
    y_batch[:, 1] = PLOT_TARGET_POSE[1]
    sols, _ = inverse_kinematics(model, y_batch, ndofs, dim_z)
    for i in range(50):
        draw_robot_on(ax, sols[i].detach().cpu().numpy(), "k")

    # Calculate average l2 error
    y_realized = forward_kinematics(sols)
    positional_errors = np.linalg.norm(y_realized.detach().cpu().numpy() - np.array(PLOT_TARGET_POSE), axis=1)
    ave_l2_error = np.mean(positional_errors)
    ax.set_title(f"50 IK solutions for y={PLOT_TARGET_POSE}")
    ax.text(-4, 4, f"global_step: {global_step}")
    ax.text(-4, 3.75, f"ave_l2err:    {round(ave_l2_error,3)}")

    plt.savefig("tempfile.png", dpi="figure", bbox_inches="tight")
    plt.close()
    return Image.open("tempfile.png")


""" We are using a 4dof planar robot. Each joint is [-pi, pi], and each link is 1m long

Example usage:

python train.py \
    --nb_nodes=6 \
    --dim_latent_space=5 \
    --coeff_fn_config=2 \
    --coeff_fn_internal_size=256 \
    --batch_size=64 \
    --learning_rate=0.001
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Wandb test")

    # Training hps
    parser.add_argument("--nb_nodes", type=int, default=6)
    parser.add_argument("--dim_latent_space", type=int, default=5)
    parser.add_argument("--coeff_fn_config", type=int, default=2)
    parser.add_argument("--coeff_fn_internal_size", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)

    args = parser.parse_args()

    hparams = {
        "nb_nodes": args.nb_nodes,
        "dim_latent_space": args.dim_latent_space,
        "coeff_fn_config": args.coeff_fn_config,
        "coeff_fn_internal_size": args.coeff_fn_internal_size,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }

    model = glow_cinn_model(hparams, NDOFS)
    dim_z = args.dim_latent_space
    print(f"Average L2 Error: {calculate_ave_l2_errror(model, NDOFS, dim_z)}")
    run_tests(model, dim_z)

    plot_solutions(model, NDOFS, dim_z, 0)

    # Initialize weights and biases logging
    tags = ["V0"]
    wandb_run = wandb.init(
        project="wandb-demo",
        notes="experiment 0",
        tags=tags,
        config=hparams,
    )
    # watch() doesn't provide useful info.
    # wandb.watch(model_wrapper.nn_model, log_freq=100, log="gradients")

    wandb_run_name = wandb.run.name

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.8e-05)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)

    # Get dataset

    for global_step in range(int(1e6)):
        # Evaluate
        if global_step % EVAL_EVERY_K == 0:
            ave_l2_error, runtime = calculate_ave_l2_errror(model, NDOFS, dim_z)
            results_dict = {"global_step": global_step, "val/ave_l2_error": ave_l2_error, "runtime": runtime}
            wandb.log(results_dict)

        # Plot and upload gt vs. returned distributions
        if global_step % LOG_PLOT_EVERY_K == 0:
            outfile = plot_solutions(model, NDOFS, dim_z, global_step)
            wandb.log(
                {
                    "global_step": global_step,
                    "target_pose": str(PLOT_TARGET_POSE),
                    f"solutions_for_target_pose": wandb.Image(outfile),
                }
            )

        # Generate data
        x = 2 * PI * torch.rand((args.batch_size, NDOFS), device=device) - PI
        y = forward_kinematics(x).to(device)

        # Perform optimization step
        model.train()
        optimizer.zero_grad()
        pad_x = 1e-5 * torch.randn(args.batch_size, dim_z - NDOFS).to(device)
        x = torch.cat([x, pad_x], dim=1)
        cond = y
        output, jac = model.forward(x, c=cond, jac=True)
        zz = torch.sum(output**2, dim=1)
        neg_log_likeli = 0.5 * zz - jac
        loss = torch.mean(neg_log_likeli)
        # Backprop the loss amd update parameters
        loss.backward()
        optimizer.step()

        # Log Training results
        if global_step > 0 and global_step % LOG_LOSS_EVERY_K == 0:
            # 'If you name your metrics "prefix/metric-name", we auto-create prefix sections.'
            results_dict = {"global_step": global_step, "train/loss": loss.item()}
            wandb.log(results_dict)

        # Step learning rate every k
        if global_step > 0 and global_step % STEP_LR_EVERY == 0:
            weight_scheduler.step()
