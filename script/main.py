import os
import argparse
import json

import torch
from dotenv import load_dotenv

# repo functions:
from data.data_loader import Data_L
from model.model import (
    Generator_256,
    Discriminator_256,
    Generator_128,
    Discriminator_128,
)
from train import train
from utils.divisor import get_closest_batch_size
from utils.generate_img import generate_noise, save_gen_img
from utils.save_log import load_from_checkpoint
import utils.make_gif
from test import test


load_dotenv()


class TrainParams:
    def __init__(self, **kwargs):
        # data parameters
        self.epoch = int(os.environ["EPOCH"]) if "EPOCH" in os.environ else 100
        self.batch_size = (
            int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else 100
        )
        self.train_size = 0

        # training parameters:
        self.disc_steps = (
            int(os.environ["DISC_STEP"]) if "DISC_STEP" in os.environ else 1
        )
        self.rate_g = (
            float(os.environ["LEARNING_RATE_GENERATOR"])
            if "LEARNING_RATE_GENERATOR" in os.environ
            else 0.0002
        )
        self.rate_d = (
            float(os.environ["LEARNING_RATE_DISCRIMINATOR"])
            if "LEARNING_RATE_DISCRIMINATOR" in os.environ
            else 0.0002
        )
        self.gradient_penalty = (
            float(os.environ["GRADIENT_PENALTY"])
            if "GRADIENT_PENALTY" in os.environ
            else 0
        )
        self.betas = (0.5, 0.999)
        self.loss = "standard"
        self.seed = None

        # saving and Logging:
        self.steps_per_log = 5
        self.steps_per_img_save = (
            int(os.environ["STEPS_PER_IMG_SAVE"])
            if "STEPS_PER_IMG_SAVE" in os.environ
            else 100
        )

        self.steps_per_checkpoint = 1000
        self.steps_per_clean = 2000
        self.telegram = False

        # testing parameters:
        self.steps_per_val = 500

        for key, val in kwargs.items():
            if key == "loss" and isinstance(val, str):
                val = "wasserstein"

            if val is not None:
                self.__dict__[key] = val

        if self.train_size % self.batch_size != 0:
            self.batch_size = get_closest_batch_size(self.train_size, self.batch_size)
            print("non divisibili, scelgo batch_size:", self.batch_size)

        self.step_per_epoch = int(self.train_size / self.batch_size)
        self.steps = self.step_per_epoch * self.epoch

    def save_params(self, path_json, name="params.json"):
        data = {}
        # Itera attraverso gli attributi della classe Esempio
        for attr, value in self.__dict__.items():
            data[attr] = value

        # Converti il dizionario in un oggetto json
        json_data = json.dumps(data)

        # Salva il json su disco
        if path_json is not None:
            if os.path.isdir(path_json):
                if os.path.isfile(os.path.join(path_json, name)) == False:
                    with open(
                        os.path.join(path_json, name),
                        "w",
                    ) as f:
                        f.write(json_data)
            else:
                os.mkdir(path_json)
                if os.path.isfile(os.path.join(path_json, name)) == False:
                    with open(
                        os.path.join(path_json, name),
                        "w",
                    ) as f:
                        f.write(json_data)


class ModelParams:
    def __init__(self, **kwargs):
        self.img_dim = 128
        self.z_dim = 100

        self.random_seed = 42

        self.generator_transpose = True
        self.discriminator_spectral_norm = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.out_dir = "OUTPUT"

        self.n_generate_img = 100

        self.telegram = True

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


def main():
    model_params = ModelParams()  # initialize model defaults parameters

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--mode", type=str, help="start training", default="train")
    parser.add_argument(
        "--out", type=str, help="out directory", default=model_params.out_dir
    )

    parser.add_argument(
        "--img_dim",
        type=int,
        help="dimension of img: 128 or 256",
        default=model_params.img_dim,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="set random seed",
        default=model_params.random_seed,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run this script: cpu or cuda",
        default=model_params.device,
    )
    parser.add_argument(
        "--telegram", help="send alert on telegram", action="store_true", default=False
    )
    parser.add_argument(
        "--gif",
        help="make gif at the end of training",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # imposto il seed se necessario altrimenti lascio casuale
    if args.random_seed == None or args.random_seed <= 0:
        pass
    else:
        torch.manual_seed(args.random_seed)

    if args.img_dim == 128:
        generator = Generator_128(transpose=model_params.generator_transpose)
        discriminator = Discriminator_128(
            spectral_norm=model_params.discriminator_spectral_norm
        )
    elif args.img_dim == 256:
        generator = Generator_256()
        discriminator = Discriminator_256()
    else:
        print("unknown img_dim:", args.img_dim)
        exit()

    if args.mode == "train":
        # create dataset
        dataset = Data_L(data_folder=os.path.join("script", "input", str(args.img_dim)))
        # take parameters
        train_params = TrainParams(
            train_size=dataset.train_size, seed=args.random_seed, telegram=args.telegram
        )

        # mi salvo i parametri nella cartella di uscita
        train_params.save_params(args.out)
        print(f"train params: {train_params.__dict__}")

        os.makedirs(args.out, exist_ok=True)
        train(
            generator,
            discriminator,
            train_dataloader=dataset.get_train_set(batch_size=train_params.batch_size),
            val_dataloader=dataset.get_val_set(batch_size=train_params.batch_size),
            params=train_params,
            out_dir=args.out,
            device=args.device,
        )

        if args.gif:
            print("saving gif...")
            utils.make_gif.make_gif(
                os.path.join(args.out, "images_output"), "gen_s", args.out
            )

    elif args.mode == "generate":
        load_from_checkpoint(
            os.path.join(os.getcwd(), "script", "model", "model.pt"),
            generator,
            discriminator,
        )
        for i in range(20):
            noise = generate_noise(1, 100, 3).cuda()
            save_gen_img(
                generator.cuda(),
                noise,
                path="",
                title=f"output_{i}.jpg",
                device=args.device,
            )
    elif args.mode == "test":
        dataset = Data_L(data_folder=os.path.join("script", "input", str(args.img_dim)))
        train_params = TrainParams(
            train_size=dataset.train_size, seed=args.random_seed, telegram=args.telegram
        )

        test(
            generator,
            discriminator,
            test_dataloader=dataset.get_test_set(batch_size=train_params.batch_size),
            out_dir="OUTPUT_TEST",
            device=args.device,
        )


if __name__ == "__main__":
    main()
