import matplotlib.pyplot as plt


def plot_matshow(A, fontsize, labelsize, cbar_label):
    fig = plt.figure(figsize=(4.5, 4))
    ax = plt.subplot(111)
    cax = ax.matshow(A, aspect='auto')
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.set_label(cbar_label, fontsize=fontsize+2, rotation=0, labelpad=15)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.show()
