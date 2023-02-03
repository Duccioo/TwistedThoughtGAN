import random as r
import matplotlib.pyplot as plt
from numpy import arange, sin, pi
import tqdm

plt.rcParams["figure.figsize"] = 6, 6  # size of plot in inches


# dpi for 128 migliore 50
# dpi for 256 migliore 40
import os


def gen_harmonograph(
    n_images=100,
    n_pend=None,
    n_steps=None,
    my_dpi=40,
    dimension=256,
    path="256",
    suffix=None,
    line_width=[0.24, 0.33],  # line width max and min
):
    for index in tqdm.tqdm(range(n_images[0], n_images[1]), smoothing=0.5):
        if n_pend == None:
            npend = r.randint(2, 4)

        if n_steps == None:
            steps = r.randrange(25000, 46000, 5000)

        mf = npend  # # of pendulums & maximum frequency, di solito con npend = 2 ottengo circonferenze

        sigma = r.uniform(0.003, 0.004)
        step = 0.0110

        if steps > 34000:
            linew = line_width[0]
        else:
            linew = line_width[1]

        t = arange(steps) * step  # time axis
        d = 1 - arange(steps) / steps  # decay vector

        ax = [r.uniform(0.5, 1.5) for i in range(npend)]
        ay = [r.uniform(0.5, 2) for item in range(npend)]

        if r.randint(0, 6) == 3:  # ancora più caos..
            ax = ay  # quando sono uguali ottengo figure chiuse

        px = [r.uniform(0, 2 * pi) for i in range(npend)]
        py = [r.uniform(0, 2 * pi) for i in range(npend)]
        fx = [r.randint(1, mf) + r.gauss(0, sigma) for i in range(npend)]
        fy = [r.randint(1, mf) + r.gauss(0, sigma) for i in range(npend)]

        if npend == 2:
            # if abs(fx[0] - fy[0]) < 00.1 and abs(fx[1] - fy[1]) < 00.1: per prevenire cerchi
            fx[0] += 1
            fx[1] += 1

        x = y = 0
        for i in range(npend):
            x += d * (ax[i] * sin(t * fx[i] + px[i]))
            y += d * (ay[i] * sin(t * fy[i] + py[i]))

        plt.figure(
            facecolor="white",
            figsize=(dimension / my_dpi, dimension / my_dpi),
            dpi=my_dpi,
        )
        plt.plot(x, y, "k", linewidth=linew)
        plt.axis("off")
        plt.gray()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

        if os.path.isdir(path) == False:
            os.mkdir(path)

        if suffix != None:
            name_harmonog = f"h_{index}" + suffix + ".png"
        else:
            name_harmonog = f"h{index}_{npend}_{steps/1000:.0f}_{sigma*10:.2f}.png"
        plt.savefig(
            os.path.join(path, name_harmonog),
            dpi=my_dpi,
        )

        plt.close()


def find_best_dpi(
    start=10, end=256, dimension=256, path="256", line_width=[0.24, 0.33], fixed=True
):
    if fixed != True:

        for i in range(start, end):
            gen_harmonograph(
                n_images=1,
                my_dpi=i,
                dimension=dimension,
                path=path,
                suffix=str(i),
                line_width=line_width,
            )

    else:
        npend = r.randint(2, 4)
        steps = r.randrange(25000, 46000, 5000)
        sigma = 0.003

        if steps > 34000:
            linew = line_width[0]
        else:
            linew = line_width[1]

        step = 0.0110

        mf = npend  # # of pendulums & maximum frequency, di solito con npend = 2 ottengo circonferenze

        t = arange(steps) * step  # time axis
        d = 1 - arange(steps) / steps  # decay vector

        ax = [r.uniform(0.5, 1.5) for i in range(npend)]
        ay = [r.uniform(0.5, 2) for item in range(npend)]

        if r.randint(0, 6) == 3:  # ancora più caos..
            ax = ay  # quando sono uguali ottengo figure chiuse

        px = [r.uniform(0, 2 * pi) for i in range(npend)]
        py = [r.uniform(0, 2 * pi) for i in range(npend)]
        fx = [r.randint(1, mf) + r.gauss(0, sigma) for i in range(npend)]
        fy = [r.randint(1, mf) + r.gauss(0, sigma) for i in range(npend)]

        if npend == 2:
            # if abs(fx[0] - fy[0]) < 00.1 and abs(fx[1] - fy[1]) < 00.1: per prevenire cerchi
            fx[0] += 1
            fx[1] += 1

        for index in tqdm.tqdm(range(start, end), smoothing=0.5):

            x = y = 0
            for i in range(npend):
                x += d * (ax[i] * sin(t * fx[i] + px[i]))
                y += d * (ay[i] * sin(t * fy[i] + py[i]))

            plt.figure(
                facecolor="white",
                figsize=(dimension / index, dimension / index),
                dpi=index,
            )
            plt.plot(x, y, "k", linewidth=linew)
            plt.axis("off")
            plt.gray()
            plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

            if os.path.isdir(path) == False:
                os.mkdir(path)

            name_harmonog = f"h{index}_{npend}_{steps/1000:.0f}_{sigma*10:.2f}.png"

            plt.savefig(
                os.path.join(path, name_harmonog),
                dpi=index,
            )

            plt.close()


def main():
    # find_best_dpi(
    #     dimension=128,
    #     path="128_dpi",
    #     start=10,
    #     end=128,
    #     line_width=[0.12, 0.24],
    #     fixed=True,
    # )
    gen_harmonograph(n_images=[0, 20000], my_dpi=50, dimension=128, path="128")


if __name__ == "__main__":
    main()
