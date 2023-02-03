import os

# torch:
import torch
import torch.nn.functional as F

# progress bar
import tqdm

# repo functions:
from utils.save_log import (
    load_from_checkpoint,
    log,
    latest_checkpoint,
    save_model,
    save_checkpoint,
)
from utils.generate_img import generate_noise
from validation import validation


_checkpoint_base_name = "state_dict"


def train(
    generator,
    discriminator,
    train_dataloader,
    val_dataloader,
    params,
    checkpoint=None,
    try_load=True,
    out_dir="OUTPUT_training",
    device="cpu",
):
    # inizializzo le metriche per la fase di validation
    psnr = 0
    ssim_v = 0
    accuracy = 0
    recall = 0
    precision = 0
    f1_score = 0
    support = 0
    swd = 0

    step = 0
    g_step = 0
    epochs_saved = 0
    g_loss = torch.tensor(0).to(device)

    imgs_dir = os.path.join(out_dir, "images_output")
    checkpints_dir = os.path.join(out_dir, "checkpoint")

    for epoch in tqdm.tqdm(
        range(epochs_saved, params.epoch), smoothing=0.6, position=0
    ):
        tqdm_gen = tqdm.tqdm(train_dataloader, smoothing=0.7, position=1, leave=False)

        for real_samples, _ in tqdm_gen:

            # D step
            z = generate_noise(
                real_samples.size(0), dim=100, n_distributions=3, device=device
            )
            fake_samples = generator(z)
            optimizer_D.zero_grad()
            d_loss = discriminator_loss(
                discriminator, real_samples.to(device), fake_samples, params
            )
            d_loss.backward()
            optimizer_D.step()

            # G step
            if step % params.disc_steps == 0:
                optimizer_G.zero_grad()
                fake_samples = generator(
                    generate_noise(real_samples.size(0), 100, 3, device=device)
                )
                g_loss = generator_loss(discriminator, fake_samples, params.loss)
                g_loss.backward()
                optimizer_G.step()

                g_step += 1

            if step % params.steps_per_val == 0 and step > 0:

                (
                    psnr,
                    ssim_v,
                    swd,
                    accuracy,
                    recall,
                    precision,
                    f1_score,
                    support,
                ) = validation(
                    val_dataloader,
                    generator,
                    discriminator,
                    keep_training=True,
                    device=device,
                )

            log(
                out_dir=out_dir,
                generator=generator,
                discriminator=discriminator,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                d_loss=d_loss,
                g_loss=g_loss,
                params=params,
                g_step=g_step,
                imgs_dir=imgs_dir,
                progress_bar=tqdm_gen,
                run_epoch=epoch,
                run_step=step,
                checkpoint_base_name=_checkpoint_base_name,
                checkpoints_dir=checkpints_dir,
                psnr=psnr,
                ssim=ssim_v,
                accuracy=accuracy,
                recall=recall,
                precision=precision,
                f1_score=f1_score,
                support=support,
                swd=swd,
                device=device,
            )

            step += 1
