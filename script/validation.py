import torch
import tqdm

# https://github.com/VainF/pytorch-msssim
from pytorch_msssim import ssim

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import normalize

from sklearn.metrics import classification_report
from utils.swd import swd
from utils.generate_img import generate_noise


def evaluate_discriminator(
    discriminator, real_data_loader, generator, device="cpu", keep_training=True
):
    discriminator.eval().to(device)
    generator.eval().to(device)

    predictions = []
    labels = []
    with torch.no_grad():
        for real_data, _ in tqdm.tqdm(real_data_loader, leave=False):
            real_data = real_data.to(device)

            real_labels = torch.ones(real_data.size(0), 1, device=device)
            fake_labels = torch.zeros(real_data.size(0), 1, device=device)
            # genero fake data
            noise = generate_noise(
                batch_size=real_data.size(0), dim=100, n_distributions=3, device=device

            )
            fake_data = generator(noise.to(device)).detach()

            output = discriminator(real_data)
            predictions += list(((output)).long().cpu().numpy())
            labels += list(real_labels.cpu().numpy())

            output = discriminator(fake_data)
            predictions += list(((output)).long().cpu().numpy())
            labels += list(fake_labels.cpu().numpy())

    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )

    if keep_training:
        discriminator.train().to(device)
        generator.train().to(device)

    return (
        report["accuracy"],
        report["macro avg"]["recall"],
        report["macro avg"]["precision"],
        report["macro avg"]["f1-score"],
        report["macro avg"]["support"],
    )


def evaluate_generator(generator, real_data_loader, device="cpu", keep_training=True):

    generator.eval().to(device)
    # fid = fid_score(generator=generator, dataloader=real_data_loader, device=device)
    psnr = 0
    ssim_value = 0
    swd_value = 0
    wd_value = 0
    with torch.no_grad():
        for img, _ in tqdm.tqdm(real_data_loader, leave=False):
            noise = generate_noise(batch_size=img.size(0), dim=100, n_distributions=3, device=device)
            fake_img = generator(noise.to(device))
            swd_value += swd(fake_img, img.to(device), device=device)
            psnr += calculate_psnr(fake_img, img.to(device))
            ssim_value += calculate_ssim(fake_img.to(device), img.to(device))

        swd_value = swd_value / len(real_data_loader)
        psnr = psnr / len(real_data_loader)
        ssim_value = ssim_value / len(real_data_loader)

    if keep_training:
        generator.train().to(device)

    return psnr.item(), ssim_value.item(), swd_value.item()


def fid_score(generator, dataloader, device="cpu", batch_size=10):

    """
    Calcola il FID score utilizzando un generatore e un dataloader di immagini reali.
    """
    # Carica la rete Inception-v3
    inception = models.inception_v3(pretrained=True)
    inception.eval()
    # Rimuovi la parte di classificazione
    inception = nn.Sequential(*list(inception.children())[:-1])
    # Definisci una funzione per calcolare le feature
    def get_features(x):
        x = inception(x)
        x = normalize(x, p=2, dim=1)
        return x.view(x.size(0), -1)

    # Calcola le feature delle immagini reali
    real_features = []
    for img, _ in dataloader:
        img = img.to(device)
        real_features.append(get_features(img).cpu())
    real_features = torch.cat(real_features, dim=0)
    # Calcola le feature delle immagini generate
    generator.eval()
    generator = generator.to(device)
    noise = torch.randn(batch_size, 100).to(device)
    fake_imgs = generator(noise)
    fake_features = get_features(fake_imgs)
    # Calcola la distanza tra le distribuzioni delle feature
    m_real = real_features.mean(0)
    m_fake = fake_features.mean(0)
    s_real = real_features.std(0)
    s_fake = fake_features.std(0)
    mean_distance = torch.norm(m_real - m_fake)
    std_distance = torch.norm(s_real - s_fake)
    fid_score = mean_distance + std_distance
    return fid_score


def calculate_ssim(img1, img2, data_range=1):

    return ssim(img1, img2, data_range=data_range)


def calculate_psnr(gen_images, real_images):
    mse = torch.mean((gen_images - real_images) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr


def validation(
    dataloader,
    generator,
    discriminator,
    device="cpu",
    keep_training=True,
):

    accuracy, recall, precision, f1_score, support = evaluate_discriminator(
        discriminator,
        generator=generator,
        real_data_loader=dataloader,
        device=device,
        keep_training=keep_training,
    )

    psnr_value, ssim_value, swd_value = evaluate_generator(
        generator,
        real_data_loader=dataloader,
        device=device,
        keep_training=keep_training,
    )

    return (
        psnr_value,
        ssim_value,
        swd_value,
        accuracy,
        recall,
        precision,
        f1_score,
        support,
    )


if __name__ == "__main__":
    pass
