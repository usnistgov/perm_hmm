import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)
import fire
import numpy as np
import matplotlib.pyplot as plt
from example_systems.beryllium import dimensionful_gamma, l_to_fmf
from example_systems import beryllium




def plot_tmats(
        data_directory=None,
):
    min_time = 5.5e-7
    max_time = 5.39e-5
    first_mat = beryllium.log_transition_matrix(min_time * dimensionful_gamma)
    last_mat = beryllium.log_transition_matrix(max_time * dimensionful_gamma)
    if data_directory is None:
        data_directory = os.getcwd()
    filename = os.path.join(data_directory, 'tmats.npz')
    np.savez(filename, first_mat=first_mat, last_mat=last_mat)

    fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2)
    vmin = None
    for ax, mat, time, label, location in zip([ax1, ax2], [first_mat, last_mat],
                                              [min_time, max_time],
                                              ["(a)", "(b)"], [-.1, 1.15]):
        mat = mat / np.log(10)
        mat[mat < -10] = -np.inf
        if vmin is None:
            vmin = mat.min()
        matimg = ax.matshow(mat, vmax=0)
        ax.set_yticks(np.arange(8))
        ax.set_yticklabels([l_to_fmf[x] for x in range(8)])
        ax.tick_params(axis=u'y', which=u'both', length=0)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels([l_to_fmf[x] for x in range(8)])
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")
        ax.text(location, 1.15, label, transform=ax1.transAxes, fontsize=14,
                va='top', ha='right')
        ax.set_title(
            "$\\Delta t = {:.2f} \\mu\\mathrm{{s}}$\n$j$".format(time * 1e6))
    cb = plt.colorbar(matimg, ax=[ax1, ax2], shrink=.55,
                      label=r"$\log_{10}(R(j|i))$")
    cb.ax.minorticks_on()
    filename = os.path.join(data_directory, "tmats.svg")
    plt.savefig(filename)
    plt.show()


def main():
    plt.rc('text', usetex=True)

    font = {'family': 'serif', 'size': 12, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    font = {'size': 12, 'sans-serif': ['computer modern sans-serif']}

    plt.rc('font', **font)
    plt.rcParams.update({
        'text.latex.preamble': r'\usepackage{amsfonts}',
    })

    fire.Fire(plot_tmats)



if __name__ == '__main__':
    main()