import matplotlib.pyplot as plt
def show_heatmap(map):
    #sns.heatmap(map)
    plt.colorbar()
    #plt.show()
    plt.hist(data.flatten(), bins=n_bins, density=True) # range=(0., 1.),
    #hist, bin_edges = np.histogram(data, density=True, bins=n_bins)
    #hist *= np.diff(bin_edges)    
    #plt.fill( np.arange(n_bins) / n_bins, hist)
    plt.grid(True)
    #plt.show()
    return figure