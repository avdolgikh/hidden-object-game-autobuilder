import matplotlib.pyplot as pltimport seaborn as sns
def show_heatmap(map):    figure = plt.figure(figsize=(14, 13), dpi=250)    plt.imshow(map, cmap='hot')    # cmap='viridis'
    #sns.heatmap(map)
    plt.colorbar()
    #plt.show()    return figuredef show_hist(data):    figure = plt.figure(figsize=(14, 13), dpi=250)    n_bins = 100    
    plt.hist(data.flatten(), bins=n_bins, density=True) # range=(0., 1.),
    #hist, bin_edges = np.histogram(data, density=True, bins=n_bins)
    #hist *= np.diff(bin_edges)    
    #plt.fill( np.arange(n_bins) / n_bins, hist)
    plt.grid(True)
    #plt.show()
    return figure