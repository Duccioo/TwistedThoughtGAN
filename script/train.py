import os


# torch:
import torch
import torch.nn.functional as F


# progress bar and color print
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


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device="cpu"):
    """Calculates the gradient penalty loss"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(
        [real_samples.size(0), 1, real_samples.size(2), real_samples.size(2)],
        device=device,
    )

    # Get random interpolation between real and fake samples
    interpolates = (
        alpha * real_samples + ((1.0 - alpha) * fake_samples)
    ).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generator_loss(discriminator, fake, loss_type):
    if loss_type == "standard":
        return -(torch.log(discriminator(fake))).mean()

    elif loss_type == "js":
        return (torch.log(1.0 - discriminator(fake))).mean()

    elif loss_type == "wasserstein" or loss_type == "hinge":
        return -(discriminator(fake)).mean()


def discriminator_loss(discriminator, real, fake, params, device="cpu"):
    loss = 0

    if params.loss == "standard" or params.loss == "js":
        loss = (
            -(torch.log(discriminator(real))).mean()
            - (torch.log(1 - discriminator(fake))).mean()
        )

    elif params.loss == "wasserstein":
        real_validity = discriminator(real).mean()
        fake_validity = discriminator(fake).mean()
        loss = -real_validity + fake_validity

    elif params.loss == "hinge":
        loss = (
            F.relu(1.0 - discriminator(real)).mean()
            + F.relu(1.0 + discriminator(fake)).mean()
        )

    if params.gradient_penalty > 0.0:
        loss += params.gradient_penalty * compute_gradient_penalty(
            discriminator, real.data, fake.data, device=device
        )
    return loss


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

    generator.train().to(device)
    discriminator.train().to(device)

    # optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=params.rate_g, betas=params.betas
    )
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=params.rate_d)

    # Checkpoint load
    if (checkpoint is None or ~os.path.isfile(checkpoint)) and (
        try_load and os.path.isdir(checkpints_dir)
    ):
        print(f"trying to load latest checkpoint from directory {checkpints_dir}")
        checkpoint = latest_checkpoint(checkpints_dir, _checkpoint_base_name)

    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            step, epochs_saved, g_step = load_from_checkpoint(
                checkpoint,
                generator,
                discriminator,
                optimizer_G,
                optimizer_D,
            )
            print(f"start from checkpoint at step {step} and epoch {epochs_saved}")

    # mi salvo la struttura del modello
    try:
        save_model(
            out_dir,
            "model_architecture",
            generator=generator,
            discriminator=discriminator,
            batch_size=params.batch_size,
            generator_size=100,
            discriminator_size=(
                1,
                128,
                128,
            ),
        )
    except:
        save_model(
            out_dir,
            "model_architecture",
            generator=generator,
            discriminator=discriminator,
            batch_size=params.batch_size,
            generator_size=100,
            discriminator_size=(
                1,
                256,
                256,
            ),
        )



    # Actual training
    # ---------------------------------Loop di addestramento-----------------------------------

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
                discriminator,
                real_samples.to(device),
                fake_samples,
                params,
                device=device,
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

    # alla fine della fase di test mi salvo il modello finale
    save_checkpoint(
        checkpints_dir,
        f"model_final_{step}_{epoch}.pt",
        generator,
        discriminator,
        step=step,
        epoch=epoch,
        g_step=g_step,
    )


if __name__ == "__main__":
    pass
