import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def plot_citations(y_thresh=150):
    data = pd.read_excel("/Users/elias/Library/CloudStorage/OneDrive-epfl.ch/EMTR-with-super-resolution-review.xlsx")
    citations = data["Citations"]
    citations = np.sort(citations, )
    filt = []
    for di in data["Note"]:
        if not isinstance(di, str):
            filt.append(False)
            continue
        if "does not qualify" in di.lower():
            filt.append(False)
        else:
            filt.append(True)

    citations = citations[filt]
    citations_pos = citations[citations > 0]
    plt.figure(figsize=(5, 4))
    plt.stairs(np.log10(citations_pos[::-1]), color="k", label="Citations")
    plt.ylim(-.5, 3)

    plt.hlines(np.log10(y_thresh), 0, citations.size, colors="red", linestyles="--", label="Cutoff")

    plt.legend()

    plt.xlim(0, citations.size)
    plt.xlabel("Paper")
    plt.ylabel("Log$_{10}$ of number of citations")
    plt.title("Distribution of Crossref citations")
    plt.tight_layout()
    plt.savefig("citations.pdf")


def plot_resolution():
    data = pd.read_excel("/Users/elias/Library/CloudStorage/OneDrive-epfl.ch/EMTR-with-super-resolution-review.xlsx")
    dates = data["Publication date"]
    resolutions = data["Resolution"]
    resolutions_usable = []
    dates_usable = []
    pattern = re.compile(r'lambda/(\d*\.?\d+)')
    print(resolutions[:10])
    for res, date in zip(resolutions, dates):
        if not isinstance(res, str):
            continue
        m = pattern.search(res)
        if m:
            num_str = m.group(1)
            # convert to int if no decimal point, else float
            x = int(num_str) if '.' not in num_str else float(num_str)
            print(f"{res!r} → {x!r} ({type(x).__name__})")
            dates_usable.append(date)
            resolutions_usable.append(x)
        else:
            print(f"{res!r} → no match")

    plt.figure()

    plt.plot(dates_usable, resolutions_usable, "k.")
    plt.xlabel("Publication date")
    plt.ylabel("Resolution ($\lambda^{-1}$)")

    plt.title("Published resolution over years")

    plt.tight_layout()
    plt.savefig("resolution_over_time.pdf")


def plot_he_2020():
    data_b = [
        -449.12706110572265, 0.015527950310559202,
        -302.03685741998066, 0.034161490683229934,
        -199.03006789524738, 0.07142857142857162,
        -151.0184287099903, 0.11697722567287805,
        -98.64209505334628, 0.23291925465838526,
        -50.63045586808926, 0.5476190476190477,
        -27.06110572259945, 0.8022774327122153,
        -9.602327837051462, 0.9710144927536232,
        0.4364694471387338, 0.9968944099378882,
        10.475266731328816, 0.9679089026915114,
        50.19398642095041, 0.5600414078674952,
        100.38797284190105, 0.23913043478260876,
        149.27255092143548, 0.1252587991718428,
        201.21241513094083, 0.07453416149068337,
        301.16391852570325, 0.035196687370600666,
        450.43646944713873, 0.019668737060041463,
    ]
    data_d = [
        -449.5635305528613, 0.006211180124223725,
        - 303.7827352085354, 0.012422360248447228,
        - 178.95247332686716, 0.021739130434782705,
        - 122.21144519883609, 0.048654244306418404,
        - 97.76915615906893, 0.06107660455486563,
        - 81.18331716779824, 0.10662525879917184,
        - 51.50339476236661, 0.18530020703933747,
        - 42.77400581959262, 0.2474120082815735,
        - 32.2987390882638, 0.3571428571428572,
        - 27.934044616876804, 0.4296066252587992,
        - 23.569350145489807, 0.5434782608695652,
        - 14.839961202715813, 0.7525879917184266,
        - 9.602327837051462, 0.8581780538302277,
        - 4.364694471386997, 0.9472049689440994,
        - 1.1368683772161603e-13, 0.9968944099378882,
        6.983511154219173, 0.8975155279503106,
        14.403491755577079, 0.7546583850931677,
        25.75169738118325, 0.47515527950310565,
        38.40931134820562, 0.30952380952380965,
        30.55286129970898, 0.3902691511387164,
        49.75751697381179, 0.18944099378881996,
        78.56450048496606, 0.11283643892339557,
        103.00678952473322, 0.06314699792960665,
        140.5431619786615, 0.038302277432712195,
        178.9524733268671, 0.025879917184264967,
        230.45586808923372, 0.01966873706004124,
        272.3569350145491, 0.019668737060041463,
        341.3191076624637, 0.019668737060041463,
        398.9330746847721, 0.01863354037267073,
        449.99999999999955, 0.01863354037267073,
    ]

    x_b, rho_b = np.array(data_b[::2]), np.array(data_b[1::2])
    ind = np.argsort(x_b)
    x_b, rho_b = x_b[ind], rho_b[ind]
    x_d, rho_d = np.array(data_d[::2]), np.array(data_d[1::2])
    ind = np.argsort(x_d)
    x_d, rho_d = x_d[ind], rho_d[ind]
    f_b = 37.3e3
    f_d = 87.6e3

    gamma_s = -.96
    gamma_t = .99
    alpha = 0
    length = 0

    delta_tau = -np.log(np.abs(gamma_s * gamma_t)) + 2 * alpha * length

    print(delta_tau / 2 / np.pi * 1e2)

    plt.figure(figsize=(4, 3))

    plt.plot(x_b / 3e8 * f_b * 1e2, rho_b, "r", label="$f_1$")
    plt.plot(x_d / 3e8 * f_d * 1e2, rho_d, "k--", label="$f_2$")
    plt.legend()
    plt.xlim(-5, 5)
    plt.xlabel(r"$x/\lambda\times 100$")
    plt.title("Correlation")
    plt.tight_layout()
    plt.savefig("he_2020.pdf")


def main():
    plot_he_2020()
    plot_citations()
    plot_resolution()
    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.use("TkAgg")

    main()
