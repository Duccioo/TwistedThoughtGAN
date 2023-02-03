import os
import re
from PIL import Image
import imageio


def make_gif(root_dir, prefix, root_out=None):
    # crea una gif da una serie di immagini redisenti nella cartella root_dir

    files = [
        f"{root_dir}/{f}" for f in os.listdir(root_dir) if re.match(f"{prefix}*", f)
    ]

    files.sort(
        key=lambda x: int(
            x.split(".")[0].strip(root_dir).strip("/" + prefix).split("_")[0]
        )
    )

    project_name = root_dir.split(os.sep)[-2]

    out_name = os.path.join(root_out, str(project_name) + ".gif")

    imageio.mimsave(out_name, [Image.open(f) for f in files])


if __name__ == "__main__":
    make_gif(os.path.join("", "images_output"), "gen_s", "")
