import itertools, os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import shutil
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import warnings
import json
from accelerate import PartialState


# Validation has been moved to a separate script: run_validation_monitor.py
# The run_validation_pass function has been removed from here.
def train_loop(
    logger,
    args,
    initial_global_step,
    first_epoch,
    accelerator,
    train_dataloader,
    unet,
    text_encoder,
    vae,
    noise_scheduler,
    weight_dtype,
    optimizer,
    lr_scheduler,
    ema_unet,
    hcn=None,  # Hierarchical Conditioner Network for compositional demographics
):
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = initial_global_step
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info("Epoch {}, global step {}".format(epoch, global_step))

        if args.train_text_encoder:
            text_encoder.train()

        # Set HCN to training mode
        if hcn is not None:
            hcn.train()

        for step, batch in enumerate(train_dataloader):
            logger.info("*** batch {} ***".format(batch["pixel_values"].shape))
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.image_type == "pt":
                    # VAE is frozen, so set to eval mode to save memory (no dropout, no batch norm updates)
                    vae.eval()
                    with torch.no_grad():  # No gradients needed for VAE encoding
                        # VAE runs in fp32 for numerical stability, accepts fp32 input
                        latents = vae.encode(
                            batch["pixel_values"].to(dtype=torch.float32)
                        ).latent_dist.sample()
                        # Legacy fudge factor from original dreambooth code
                        latents = latents * 0.18215  # vae.config.scaling_factor
                    # Cast latents to weight_dtype for UNet (this is the only dtype conversion needed)
                    latents = latents.to(dtype=weight_dtype)
                elif args.image_type == "parameters":
                    latents = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    raise Exception("not supported")
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                text_input_ids = batch["input_ids"]
                if args.use_attention_mask:
                    attention_mask = batch["attention_mask"]
                else:
                    attention_mask = None

                prompt_embeds = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                encoder_hidden_states = prompt_embeds[0]  # [B, 77, d_ctx]

                # === HCN: Hierarchical Conditioning ===
                kl_loss = None
                comp_loss = None
                aux_loss = None
                hcn_ctx_norm = None
                if hcn is not None:
                    # Get HCN demographic context
                    hcn_ctx, mu, logsigma, aux_logits = hcn(
                        batch["age_idx"],
                        batch["sex_idx"],
                        batch["race_idx"],
                    )  # hcn_ctx: [B, 1, d_ctx]

                    # Concatenate text and demographic contexts
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, hcn_ctx], dim=1
                    )  # [B, 78, d_ctx] = 77 text tokens + 1 demographic token

                    hcn_ctx_norm = hcn_ctx.norm(dim=-1).mean()

                    # Compute KL divergence loss (uncertainty regularization)
                    # KL(N(mu, sigma) || N(0, 1))
                    kl_loss = -0.5 * torch.sum(
                        1 + 2 * logsigma - mu ** 2 - torch.exp(2 * logsigma),
                        dim=-1
                    ).mean()

                    # Compute compositional consistency loss
                    # Unwrap hcn from DDP if needed to access custom methods
                    hcn_unwrapped = accelerator.unwrap_model(hcn)
                    comp_loss = hcn_unwrapped.compute_compositional_loss(
                        batch["age_idx"],
                        batch["sex_idx"],
                        batch["race_idx"],
                    )

                    # Auxiliary demographic classification losses
                    if aux_logits is not None:
                        age_ce = F.cross_entropy(aux_logits["age"], batch["age_idx"])
                        sex_ce = F.cross_entropy(aux_logits["sex"], batch["sex_idx"])
                        race_ce = F.cross_entropy(aux_logits["race"], batch["race_idx"])
                        aux_loss = (age_ce + sex_ce + race_ce) / 3.0

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    print(
                        "are you sure ? --> noise_scheduler.config.prediction_type == v_prediction"
                    )
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Compute instance loss
                loss = F.mse_loss(
                    noise_pred.float(), target.float(), reduction="none"
                ).mean([1, 2, 3])
                loss_weights = batch["loss_weights"].to(dtype=weight_dtype)
                loss = (loss * loss_weights).sum() / loss_weights.sum()

                # === Add HCN losses ===
                if kl_loss is not None:
                    # Anneal KL weight from 0 to target value over training
                    kl_weight = min(1.0, global_step / args.hcn_kl_anneal_steps) * args.hcn_kl_weight
                    loss = loss + kl_weight * kl_loss

                if comp_loss is not None:
                    loss = loss + args.hcn_comp_weight * comp_loss

                if aux_loss is not None and args.hcn_aux_weight > 0:
                    loss = loss + args.hcn_aux_weight * aux_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    # Add HCN parameters to gradient clipping
                    if hcn is not None:
                        params_to_clip = itertools.chain(params_to_clip, hcn.parameters())

                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                # Save state checkpoint
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            # Convert to int in case it was loaded as string from YAML
                            try:
                                checkpoints_total_limit = int(args.checkpoints_total_limit)
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid checkpoints_total_limit value: {args.checkpoints_total_limit}. Skipping checkpoint cleanup.")
                                checkpoints_total_limit = None
                            
                            # Only proceed with cleanup if we have a valid limit
                            if checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )
                                
                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints) - checkpoints_total_limit + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            args.output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Note: Validation has been moved to a separate script (run_validation_monitor.py)
                # It monitors this directory and automatically validates new checkpoints

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            # Add HCN losses to logging
            if kl_loss is not None:
                logs["kl_loss"] = kl_loss.detach().item()
                logs["kl_weight"] = kl_weight
            if comp_loss is not None:
                logs["comp_loss"] = comp_loss.detach().item()
            if aux_loss is not None:
                logs["aux_loss"] = aux_loss.detach().item()
            if hcn_ctx_norm is not None:
                logs["hcn_ctx_norm"] = hcn_ctx_norm.detach().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    logger.info("Training finished")
    return (
        logger,
        args,
        accelerator,
        train_dataloader,
        unet,
        text_encoder,
        vae,
        noise_scheduler,
        weight_dtype,
        optimizer,
        lr_scheduler,
        ema_unet,
        progress_bar,
    )
