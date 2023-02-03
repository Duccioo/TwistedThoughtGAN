import os

# repo functions:
from utils.save_log import (
    save_validation,
)
from validation import validation


def test(
    generator,
    discriminator,
    test_dataloader,
    out_dir="OUTPUT_training",
    device="cpu",
):
    psnr = 0
    ssim_v = 0
    accuracy = 0
    recall = 0
    precision = 0
    f1_score = 0
    support = 0
    swd = 0

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
        test_dataloader,
        generator,
        discriminator,
        keep_training=True,
        device=device,
    )

    save_validation(
        accuracy,
        ssim_v,
        psnr,
        swd,
        recall,
        precision,
        f1_score,
        support,
        0,
        0,
        0,
        0,
        filename=os.path.join(out_dir, "test.csv"),
    )
