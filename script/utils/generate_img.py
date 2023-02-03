import torch
import torchvision.utils as vutils
import torch.distributions as td
import os
import os
from concurrent.futures import ProcessPoolExecutor


def delete_img_from_folder(folder_path, img_extension=("jpg", "jpeg", "png", "gif")):
    # specifica la cartella in cui si vogliono eliminare le immagini

    # utilizza ProcessPoolExecutor per eseguire operazioni in parallelo
    with ProcessPoolExecutor() as executor:
        # cicla attraverso tutti i file nella cartella
        for file_name in os.listdir(folder_path):
            # costruisce il percorso completo del file
            file_path = os.path.join(folder_path, file_name)
            # controlla se il file è un'immagine
            if file_name.endswith(img_extension):
                # invia il task di eliminazione del file al pool di processi
                executor.submit(os.remove, file_path)
        print("Eliminazione completata")


@torch.no_grad()
def generate_noise(batch_size, dim, n_distributions, seed=None, device="cpu"):
    """
    Generates noise vector using multiple gaussian distributions.

    Parameters:
        batch_size (int): the size of the batch
        dim (int): the dimension of the noise vector
        n_distributions (int): number of gaussian distributions to combine
    """
    if seed is not None:

        noise = torch.empty(batch_size, dim, device=device).uniform_(
            -1, 1, generator=torch.Generator(device=device).manual_seed(seed)
        )

        for i in range(n_distributions):
            # gaussian = td.Normal(0, 1)
            gaussian = torch.normal(
                0,
                1,
                size=(batch_size, dim),
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
            )
            noise += gaussian

        

    else:
        noise = torch.empty(batch_size, dim, device=device).uniform_(
            -1,
            1,
        )

        for i in range(n_distributions):
            gaussian = td.Normal(0, 1)
            noise += gaussian.sample(torch.Size([batch_size, dim])).to(device)


    return noise


@torch.no_grad()
def save_gen_img(generator, noise, n_img=1, path="", title="output", device="cpu"):

    """
    Generates image/images and save them to disk.

    Parameters:
        generator (torch.nn.Module): the generator model
        noise (torch.Tensor): the noise vector
        n_img (int): the number of images to generate
        path (str): the path to save the generated images
        title (str): the title of the generated images
    """
    # controllo se il path è valido
    if path != "" and not os.path.exists(path):
        os.mkdir(path)

    is_training = generator.training
    generator.eval()
    noise = noise.cuda()
    generated_images = generator(noise)
    
    specific_path = os.path.join(path, title)

    # Save generated image
    vutils.save_image(generated_images.data, specific_path, normalize=True)
    generator.train(is_training)
    
    return specific_path
