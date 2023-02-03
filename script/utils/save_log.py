import torch
import os
import csv
from datetime import datetime
import json

from torchinfo import summary

from utils.generate_img import save_gen_img, generate_noise
import utils.telegram_alert

_checkpoint_base_name = "checkpoint_"


# ------------------------------------SAVING & LOADING-----------------------------------------
def load_from_checkpoint(
    checkpoint_path, generator, discriminator, optimizer_G=None, optimizer_D=None
):
    """
    Load model from checkpoint, return the step, epoch and generator step saved in the file.

    Parameters:
        - checkpoint_path (str): path to checkpoint
        - generator (nn.Module): generator model
        - discriminator (nn.Module): discriminator model
        - optimizer_G (torch.optim): optimizer for generator
        - optimizer_D (torch.optim): optimizer for discriminator
    """
    data = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(data["generator"])
    discriminator.load_state_dict(data["discriminator"])

    if (optimizer_G is not None) and (optimizer_D is not None):
        optimizer_G.load_state_dict(data["optimizer_G"])

        optimizer_D.load_state_dict(data["optimizer_D"])

    return (data["step"], data["epoch"], data["g_step"])


def save_checkpoint(
    checkpoint_path,
    checkpoint_name,
    generator,
    discriminator,
    step,
    epoch,
    optimizer_G=None,
    optimizer_D=None,
    g_step=0,
):
    """
    Save model to a checkpoint file.
    Parameters:
        - checkpoint_path (str): path to checkpoint
        - checkpoint_name (str): name of checkpoint
        - generator (nn.Module): generator model
        - discriminator (str): discriminator
        - step (int): current step
        - epoch (int): current epoch
        - optimizer_G (torch.optim): optimizer for generator
        - optimizer_D (torch.optim): optimizer for discriminator
        - g_step (int): current generator step
    """

    state_dict = {
        "step": step,
        "epoch": epoch,
        "g_step": g_step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }
    if (optimizer_G is not None) and (optimizer_D is not None):
        state_dict.update(
            {
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
            }
        )
    if os.path.exists(checkpoint_path):
        torch.save(state_dict, os.path.join(checkpoint_path, checkpoint_name))

    else:
        os.mkdir(checkpoint_path)
        torch.save(state_dict, os.path.join(checkpoint_path, checkpoint_name))


def latest_checkpoint(root_dir, base_name):
    """
    Find the latest checkpoint in a directory.
    Parameters:
        - root_dir (str): root directory
        - base_name (str): base name of checkpoint
    """
    checkpoints = [chkpt for chkpt in os.listdir(root_dir) if base_name in chkpt]
    if len(checkpoints) == 0:
        return None
    latest_chkpt = ""
    latest_step = -1
    latest_epoch = -1
    for chkpt in checkpoints:
        step = torch.load(os.path.join(root_dir, chkpt))["step"]
        epoch = torch.load(os.path.join(root_dir, chkpt))["epoch"]
        if step > latest_step or (epoch > latest_epoch):
            latest_epoch = epoch
            latest_chkpt = chkpt
            latest_step = step
    return os.path.join(root_dir, latest_chkpt)


def clean_old_checkpoint(folder_path, percentage):
    """
    Delete old checkpoints from a directory with a given percentage.
    Parameters:
        - folder_path (str): path to directory where checkpoints are stored
        - percentage (float): percentage of checkpoints to delete

    """

    # ordina i file per data di creazione
    files = [
        (f, os.path.getctime(os.path.join(folder_path, f)))
        for f in os.listdir(folder_path)
    ]
    files.sort(key=lambda x: x[1])

    # calcola il numero di file da eliminare
    files_to_delete = int(len(files) * percentage)

    # cicla attraverso i file nella cartella
    for file_name, creation_time in files[:-files_to_delete]:
        # costruisce il percorso completo del file
        file_path = os.path.join(folder_path, file_name)
        # elimina il file
        if file_name.endswith(".pt"):
            os.remove(file_path)


def save_model(
    path,
    name,
    generator,
    discriminator,
    batch_size,
    generator_size,
    discriminator_size,
):
    """
    Using summary function from torchinfo save the model architecture in a json file.

    Parameters:
        - path: path to save the model architecture
        - name: name of the file to save
        - generator: generator model
        - discriminator: discriminator model
        - batch_size: batch size, parameter for function summary
        - generator_size: generator size, parameter for function summary
        - discriminator_size: discriminator size, parameter for function summary
    """

    # create new dictionary
    models = {"generator": {}, "discriminator": {}}

    # prima mi carico il generatore
    gen_sum = summary(
        generator,
        input_size=(batch_size, generator_size),
        verbose=0,
        col_names=["output_size"],
    )
    gen_arch = str(gen_sum).split(
        "================================================================="
    )[2]

    gen_size = str(gen_sum).split(
        "================================================================="
    )[4]

    for line in gen_arch.split("├─"):
        line.split("                  ")
        models["generator"][line.split("                  ")[0].strip()] = line.split(
            "                  "
        )[1].strip()

    for line in gen_size.split("\n"):
        try:
            models["generator"][line.split(":")[0]] = float(line.split(":")[1])
        except:
            pass

    # passo al discriminatore
    disc_sum = summary(
        discriminator,
        input_size=((batch_size,) + discriminator_size),
        verbose=0,
        col_names=["output_size"],
    )
    disc_arch = str(disc_sum).split(
        "================================================================="
    )[2]

    disc_size = str(disc_sum).split(
        "================================================================="
    )[4]

    for line in disc_arch.split("├─"):
        line.split("                  ")
        models["discriminator"][
            line.split("                  ")[0].strip()
        ] = line.split("                  ")[1].strip()

    for line in disc_size.split("\n"):
        try:
            models["discriminator"][line.split(":")[0]] = float(line.split(":")[1])
        except:
            pass

    if os.path.isdir(path) == False:
        os.mkdir(path)

    with open(os.path.join(path, name + ".json"), "w") as f:
        json.dump(models, f)


def save_validation(
    accuracy,
    ssim,
    psnr,
    swd,
    recall,
    precision,
    f1_score,
    support,
    step=0,
    epoch=0,
    loss_d=0,
    loss_g=0,
    filename="metrics.csv",
):
    now = datetime.now()
    if loss_d != 0:
        loss_d_s = loss_d.item()
    else:
        loss_d_s = loss_d

    if loss_g != 0:
        loss_g_s = loss_g.item()

    else:
        loss_g_s = loss_g

    metrics = [
        step,
        epoch,
        f"{loss_d_s:.4f}",
        f"{loss_g_s:.4f}",
        f"{ssim:.4f}",
        f"{psnr:.4f}",
        f"{swd:.4f}",
        f"{accuracy:.4f}",
        f"{precision:.4f}",
        f"{recall:.4f}",
        f"{f1_score:.4f}",
        f"{support:.4f}",
        now.strftime("%Y-%m-%d %H:%M:%S"),
    ]

    header = [
        "Step",
        "Epoch",
        "Loss D",
        "Loss G",
        "SSIM",
        "PSNR",
        "SWD",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_score",
        "Support",
        "TIME",
    ]
    if os.path.exists(filename) == False:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(metrics)
    else:
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(metrics)


# ------------------------------------LOGGING------------------------------------
@torch.no_grad()
def update_log_training(progress_bar, epoch, step, total_step, d_loss, g_loss):
    # update progress bar for display the current training process information
    progress_bar.set_description(
        "step: {}/{}, epoch: {}, d_loss: {:.4f}, g_loss: {:.4f}".format(
            step, total_step, epoch + 1, d_loss, g_loss
        )
    )


def log(
    out_dir,
    generator,
    discriminator,
    optimizer_G,
    optimizer_D,
    params,
    g_step,
    progress_bar,
    checkpoints_dir="",
    imgs_dir="output",
    d_loss=0,
    g_loss=0,
    run_epoch=0,
    run_step=0,
    checkpoint_base_name=_checkpoint_base_name,
    accuracy=0,
    ssim=0,
    swd=0,
    recall=0,
    precision=0,
    psnr=0,
    f1_score=0,
    support=0,
    device="cpu",
):
    if (
        run_step % params.steps_per_log == 0
        or run_step % (params.step_per_epoch + 1) == 0
        or run_step == 0
    ):
        update_log_training(
            progress_bar, run_epoch, run_step, params.steps, d_loss, g_loss
        )

    if run_step % params.steps_per_img_save == 0 and run_step > 0:
        # ogni tot mi salvo anche un immagine
        img_path = save_gen_img(
            generator=generator,
            noise=generate_noise(3, 100, 3, seed=1599, device=device),
            path=imgs_dir,
            title=(
                f"gen_s{str(run_step)}_e{(run_epoch)}_gl{(g_loss.item()):.2f}_dl{(d_loss.item()):.2f}.jpg"
            ),
        )

        # controllo se il parametro di telegram è stato impostato
        if params.telegram == True:
            # se è attivo invio un messaggio dal bot
            utils.telegram_alert.alert(
                "TRAINING",
                img_path,
                run_step,
                params.steps,
                f"Gen Loss: <b>{(g_loss.item()):.2f}</b>\tDisc Loss: <b>{(d_loss.item()):.2f}</b>\nEpoca: <b>{(run_epoch)}</b>\tSTEP: <b>{(run_step)}</b>",
            )

    if run_step == params.steps:
        # quando finisco il training
        save_checkpoint(
            checkpoints_dir,
            f"model_final_{run_step}_{run_epoch}.pt",
            generator,
            discriminator,
            step=run_step,
            epoch=run_epoch,
            g_step=g_step,
        )

    if run_step % params.steps_per_checkpoint == 0 and run_step > 0:
        # ogni tot mi salvo il checkpoint anche degli optimizer

        save_checkpoint(
            checkpoints_dir,
            f"{checkpoint_base_name}_s{run_step}_e{run_epoch}.pt",
            generator,
            discriminator,
            step=run_step,
            epoch=run_epoch,
            g_step=g_step,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
        )

    if run_step % params.steps_per_val == 0 and run_step > 0:
        # save validation
        save_validation(
            accuracy,
            ssim,
            psnr,
            swd,
            recall,
            precision,
            f1_score,
            support,
            run_step,
            run_epoch,
            d_loss,
            g_loss,
            filename=os.path.join(out_dir, "validation.csv"),
        )

    if run_step % params.steps_per_clean == 0 and run_step > 0:
        # cancello un po di checkpoint se sono troppo vecchi
        clean_old_checkpoint(checkpoints_dir, 0.5)
