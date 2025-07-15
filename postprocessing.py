import numpy as np
import json
import matplotlib.pyplot as plt


def get_data_comosol(filename, radius=2, sep_start=0.02, sep_stop=2, d_sep=0.02):
    data = np.genfromtxt(filename, skip_header=8, )
    n_sep = data.shape[1]
    sep = np.arange(sep_start, sep_stop + d_sep, d_sep)
    x = np.linspace(-radius, radius, data.shape[0])

    e_field_norm = data[:, 1:]

    return x, sep, e_field_norm


def get_data(filename, **kwargs):
    with open(filename, 'r') as f:
        data = json.load(f)
    data = np.array(data)
    x = np.unique(data[:, 0])
    sep = np.unique(data[:, 1])
    e_field_norm = data[:, -1].reshape((x.size, sep.size))
    return x, sep, e_field_norm


def do_plot(filename, plot_max=True, **kwargs):
    x, sep, e_field_norm = get_data(filename, **kwargs)
    max_for_sep = np.max(e_field_norm, axis=0)
    e_field_norm = e_field_norm / max_for_sep[None, :]
    plt.contourf(
        x, sep, e_field_norm.T,
        cmap="jet",
        levels=21,
    )
    plt.colorbar(ticks=np.linspace(0, 1, 11))

    plt.contour(
        x, sep, e_field_norm.T,
        levels=(.5, ),
        colors="black"
    )
    e_up = e_field_norm[x >= 0, :]
    e_do = e_field_norm[x <= 0, :]
    max_up = np.argmax(e_up, axis=0)
    max_do = np.argmax(e_do, axis=0)
    x_up = x[max_up]
    x_do = x[max_do]

    if plot_max:
        plt.plot(x_up + 2, sep, "k--")
        plt.plot(x_do, sep, "k--")

    plt.plot(sep / 2, sep, "w-")
    plt.plot(-sep / 2, sep, "w-")
    plt.xlabel(r"$x/\lambda$")
    plt.ylabel(r"Source separation $d/\lambda$")
    plt.xlim(-1, 1)
    plt.tight_layout()

    plt.savefig(f"{filename.split('/')[-1]}.pdf")


def main():
    do_plot("data/1d.json")
    plt.figure()
    do_plot("data/2d.json")
    plt.figure()
    do_plot("data/3d.json")
    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.use("TkAgg")
    main()
