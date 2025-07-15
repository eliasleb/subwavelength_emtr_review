import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(f_lims_megahertz=(0, np.inf)):
    data_name = "data26"
    data = np.genfromtxt(f"data/{data_name}.txt",
                         skip_header=5)
    print(data.shape)

    lens, lens_y1, r_trm, x, f = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    x, f = np.unique(x), np.unique(f)

    if data.shape[1] > 7:
        s21 = []
        if data.shape[1] == 16 or data.shape[1] == 24:
            i_start = 6
        else:
            i_start = 5
        for i in range(i_start, data.shape[1], 2):
            s21.append(data[:, i] + 1j * data[:, i + 1])
        s21 = np.array(s21)

    # s21 = np.reshape(s21, (lens.size * lens_y1.size * r_trm.size, x.size, s21.shape[0], f.size))
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        "-d",
        "-o",
        "-*",
        "-x"
    ]
    n_plots = 6
    cmap = plt.get_cmap("plasma", n_plots)

    data_length = f.size * x.size
    for ind_src in range(x.size):
        print(f"{x[ind_src]=:.3f}")

        plt.figure()
        i_plot = 0
        for i_data in range(0, s21.shape[1], data_length):
            s21i = s21[:, i_data:i_data + data_length]
            s21i = np.reshape(s21i, (s21i.shape[0], x.size, f.size))
            lens_i, lens_y1_i, r_trm_i = int(data[i_data, 0]), data[i_data, 1], data[i_data, 2]
            label = f"lens={np.round(lens_i):.0f}, {lens_y1_i=:.2f}, {r_trm_i=:.2f}"

            calibration = np.sum(np.abs(s21i) ** 2, axis=(0, -1)) ** .5
            np.savetxt(f"data/calibration_{i_data}.txt", calibration)

            dt = s21i[:, ind_src, :][:, None, :] + s21i[:, x.size - ind_src - 1, :][:, None, :]
            tr = np.sum(
                np.abs(
                    np.sum(
                        s21i * np.conjugate(dt), axis=0
                    )
                ) ** 2, axis=-1) ** .5 / calibration
            tr_plot = tr / np.max(tr)
            plt.plot(x*100, tr_plot, line_styles[i_plot], color=cmap(i_plot), label=label)
            # plt.plot(-x*100, tr_plot, line_styles[i_plot], color=cmap(i_plot))
            i_plot += 1

        plt.xlabel("% λ0")
        plt.ylabel("(1)")
        plt.title("Normalized time-reversed energy")
        plt.xlim(-45, 45)
        # plt.ylim(.75, 1.01)
        plt.title(f"src pos = {x[ind_src]:.2f}")

        plt.tight_layout()
        plt.legend()
        plt.savefig(f"figs/{data_name}_xi_{ind_src}.pdf")

        plt.figure()
        i_plot = 0
        for i_data in range(0, s21.shape[1], data_length):
            s21i = s21[:, i_data:i_data + data_length]
            s21i = np.reshape(s21i, (s21i.shape[0], x.size, f.size))
            lens_i, lens_y1_i, r_trm_i = int(data[i_data, 0]), data[i_data, 1], data[i_data, 2]
            label = f"lens={np.round(lens_i):.0f}, {lens_y1_i=:.2f}, {r_trm_i=:.2f}"

            if i_plot > 3:
                n_d = 5
            else:
                n_d = 1

            plt.plot(f[::n_d], np.abs(s21i[0, ind_src, ::n_d]), line_styles[i_plot], label=label, color=cmap(i_plot))

            i_plot += 1

        plt.title(f"src pos = {x[ind_src]:.2f}")
        plt.xlim(np.min(f), np.max(f))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("(1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figs/s21_{data_name}_xi_{ind_src}.pdf")
        plt.show()


def get_combinations():
    combinations = [[], [], [], [], ]
    for lens in (0, 1):
        lens_y1_pos = (0., ) if lens == 0 else (0.05, .5, )
        for lens_y1 in lens_y1_pos:
            r_min = np.sqrt(
                (
                    0.5
                ) ** 2 + (
                    lens * (lens_y1 + .5)
                ) ** 2
            )
            for r_trm in (r_min * 1.1, 2.5):
                for x_src_mm in range(-500, 501, 50):
                    # for f_megahertz in range(200, 320, 1):
                    combinations[0].append(lens)
                    combinations[1].append(lens_y1)
                    combinations[2].append(r_trm)
                    combinations[3].append(x_src_mm / 1000.)
                    # combinations[4].append(f_megahertz)
    return combinations


def print_combinations():
    combinations = get_combinations()
    names = ("lens", "lens_y1", "r_trm", "src_x", "f")
    units = ("[]", "[m]", "[m]", "[m]", "[MHz]")
    with open("data/combinations.txt", "w") as fd:
        for ci, name, unit in zip(combinations, names, units):
            fd.write(f"""{name} " """)
            for xi in ci:
                fd.write(f"{xi} ")
            fd.write(f""" " {unit}\n""")


def main_pred(n_cases=6):
    data_name = "data22_trc_pred"
    data = np.genfromtxt(f"../../../../git_ignore/resonant metalens/{data_name}.txt",
                         skip_header=8)
    print(data.shape)

    combinations = get_combinations()
    lens, lens_y1, r_trm, x_src = combinations[0], combinations[1], combinations[2], combinations[3]
    if 2 * len(lens) + 1 != data.shape[1]:
        raise RuntimeError(f"The length of the data ({data.shape[1]}) is not compatible with the length of the "
                           f"combinations ({len(lens)}). Check :get_combinations:.")
    f = np.unique(combinations[4])
    x = np.unique(data[:, 0])
    ey_raw = data[:, 1:]
    x_src_unique = np.unique(x_src)
    ey = np.zeros((n_cases, x_src_unique.size, x.size, f.size), dtype="complex")
    s_sim = np.zeros((n_cases, x_src_unique.size, f.size, ), dtype="complex")
    data_length = 2 * f.size
    ind_case, ind_x_src = 0, 0
    for i_data in range(0, ey_raw.shape[1], data_length):
        s_re = (slice(None, ), slice(i_data, i_data + data_length, 2), )
        s_im = (slice(None, ), slice(i_data + 1, i_data + 1 + data_length, 2), )
        ey[ind_case, ind_x_src, :, :] = ey_raw[s_re] + 1j * ey_raw[s_im]
        s_sim[ind_case, ind_x_src, :] = ey[ind_case, ind_x_src, np.where(x >= x_src_unique[ind_x_src])[0][0], :]
        ind_x_src += 1
        if ind_x_src >= x_src_unique.size:
            ind_x_src = 0
            ind_case += 1
    # calibration = np.sum(np.abs(s_sim) ** 2, axis=-1)
    # for i_case in range(n_cases):
    #     for i_src, x_src_i in enumerate(x_src_unique):
#
    #         tr = s_sim[i_case, ...] * np.conjugate(s_sim[i_case, i_src, :][None, :])
    #         tr = np.sum(np.abs(tr) ** 2, axis=-1) # / calibration[i_case, :]
#
    #         plt.clf()
    #         plt.plot(x_src_unique, tr)
    #         plt.title(f"{i_case=}, src = {x_src_i:.2f}")
    #         plt.waitforbuttonpress()
    calibration = np.sum(np.abs(ey) ** 2, axis=(1, -1)) ** .5
    fom = np.sum(np.abs(ey) ** 2, axis=-1) ** .5  # / calibration[:, None, :]
    fom = fom / np.max(fom, axis=-1)[..., None]
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        "-d",
        "-o",
    ]
    cmap = plt.get_cmap("plasma", n_cases)
    for x_src_ind, x_src_i in enumerate(x_src_unique):
        plt.clf()
        for case_i, linestyle in enumerate(line_styles):
            # label = f"lens={np.round(lens[case_i]):.0f}, {lens_y1_i=:.2f}, {r_trm_i=:.2f}"
            plt.plot(x, fom[case_i, x_src_ind, :], linestyle, color=cmap(case_i))
            # plt.vlines(x_src_i, 0, 1)

        plt.waitforbuttonpress()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")

    print_combinations()

    main()
