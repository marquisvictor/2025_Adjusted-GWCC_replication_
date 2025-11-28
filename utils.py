import numpy as np 

def shift_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero
    Parameters
    ----------
    cmap : The matplotlib colormap to be altered
    start : Offset from lowest point in the colormap's range.
      Defaults to 0.0 (no lower ofset). Should be between
      0.0 and `midpoint`.
    midpoint : The new center of the colormap. Defaults to
      0.5 (no shift). Should be between 0.0 and 1.0. In
      general, this should be  1 - vmax/(vmax + abs(vmin))
      For example if your data range from -15.0 to +5.0 and
      you want the center of the colormap at 0.0, `midpoint`
      should be set to  1 - 5/(5 + 15)) or 0.75
    stop : Offset from highets point in the colormap's range.
      Defaults to 1.0 (no upper ofset). Should be between
      `midpoint` and 1.0.
    
    Returns
    -------
    new_cmap : A new colormap that has been shifted. 
    '''

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # cm.unregister_cmap(name=name)
    


    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    new_cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    
    # plt.register_cmap(cmap=new_cmap)

    return new_cmap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Function to truncate a colormap by selecting a subset of the original colormap's values
    Parameters
    ----------
    cmap : Mmatplotlib colormap to be altered
    minval : Minimum value of the original colormap to include in the truncated colormap
    maxval : Maximum value of the original colormap to include in the truncated colormap
    n : Number of intervals between the min and max values for the gradient of the truncated colormap
          
    Returns
    -------
    new_cmap : A new colormap that has been shifted. 
    '''

    import matplotlib as mpl

    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def compare_surfaces(data, var1, var2, gwr_bw_n, gwr_bw_o, savefig=None):


    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Theme
    sns.set_theme(
        style='white',
        context='talk',
        rc={
            'font.family': 'serif',
            'font.serif': ['Georgia'],
            'font.size': 13,
            'axes.titlesize': 16,
        }
    )


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    ax0, ax1 = axes

    ax0.set_title(f"Adjusted GWCC - Bandwidth: {gwr_bw_n}", pad=14)
    ax1.set_title(f"Classic GWCC  - Bandwidth: {gwr_bw_o}", pad=14)


    def build_cmap(values):
        """
        If data crosses zero, use symmetric limits around zero.
        If data is all positive, use red half of seismic.
        If data is all negative, use blue half of seismic.
        """

        vmin = float(values.min())
        vmax = float(values.max())
        cmap_full = plt.cm.seismic

        # Case 1 — crosses zero → symmetric around 0
        if vmin < 0 and vmax > 0:
            max_abs = max(abs(vmin), abs(vmax))
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            cmap = cmap_full
            return cmap, norm

        # Case 2 — all positive → truncate to red
        if vmin >= 0:
            from matplotlib.colors import Normalize
            cmap = truncate_colormap(cmap_full, 0.5, 1.0)
            norm = Normalize(vmin=vmin, vmax=vmax)
            return cmap, norm

        # Case 3 — all negative → truncate to blue
        if vmax <= 0:
            from matplotlib.colors import Normalize
            cmap = truncate_colormap(cmap_full, 0.0, 0.5)
            norm = Normalize(vmin=vmin, vmax=vmax)
            return cmap, norm

    # Build the colormaps
    cmap1, norm1 = build_cmap(data[var1].values)
    cmap2, norm2 = build_cmap(data[var2].values)

    # Plot surfaces
    data.plot(var1, ax=ax0, cmap=cmap1, norm=norm1,
              edgecolor='grey', linewidth=0.25)
    data.plot(var2, ax=ax1, cmap=cmap2, norm=norm2,
              edgecolor='grey', linewidth=0.25)

    # Cleanup 
    for ax in [ax0, ax1]:
        ax.set_axis_off()
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")



    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, wspace=-0.32)

    # Colorbar 1 
    bbox0 = ax0.get_position()
    cax1 = fig.add_axes([
        bbox0.x1 + 0.005,
        bbox0.y0 + 0.02,
        0.015,
        bbox0.height - 0.04
    ])

    # Colorbar 2 
    bbox1 = ax1.get_position()
    cax2 = fig.add_axes([
        bbox1.x1 + 0.005,
        bbox1.y0 + 0.02,
        0.015,
        bbox1.height - 0.04
    ])

    # Build scalar mappables
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    sm1.set_array([])
    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    sm2.set_array([])

    cb1 = fig.colorbar(sm1, cax=cax1)
    cb2 = fig.colorbar(sm2, cax=cax2)

    cb1.ax.tick_params(labelsize=14)
    cb2.ax.tick_params(labelsize=14)

    
    if savefig is not None:
        plt.savefig(savefig, format="tiff", dpi=600, bbox_inches="tight")

    plt.show()
    
    
def compare_stacked(data, var1, var2, gwr_bw_n, gwr_bw_o, savefig=None):

    import matplotlib.pyplot as plt
    import geopandas as gp
    import numpy as np
    import seaborn as sns

    # ---------------------------------
    # Apply Seaborn theme with Georgia
    # ---------------------------------
    sns.set_theme(
        style='white',
        context='talk',
        rc={
            'font.family': 'serif',
            'font.serif': ['Georgia'],
            'font.size': 13,
            'axes.titlesize': 16,
        }
    )
    sns.set_palette('bright')

    # ---------------------------------
    # Figure setup
    # ---------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    ax0, ax1 = axes

    ax0.set_title(f"Adjusted GWCC - Bandwidth: {gwr_bw_n}", pad=18)
    ax1.set_title(f"Classic GWCC - Bandwidth: {gwr_bw_o}", pad=18)

    # ---------------------------------
    # Colormap selection logic
    # ---------------------------------
    cmap = plt.cm.seismic

    improved_min = data[var1].min()
    improved_max = data[var1].max()
    classic_min = data[var2].min()
    classic_max = data[var2].max()
    vmin = np.min([improved_min, classic_min])
    vmax = np.max([improved_max, classic_max])

    # apply your smart colormap logic
    if (vmin < 0) and (vmax < 0):
        cmap = truncate_colormap(cmap, 0.0, 0.5)
    elif (vmin > 0) and (vmax > 0):
        cmap = truncate_colormap(cmap, 0.5, 1.0)
    else:
        cmap = shift_colormap(
            cmap,
            start=0.0,
            midpoint=1 - vmax / (vmax + abs(vmin)),
            stop=1.0
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # ---------------------------------
    # Plot surfaces
    # ---------------------------------
    common_kwargs = dict(
        cmap=sm.cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor='grey',
        linewidth=0.25
    )

    data.plot(var1, ax=ax0, **common_kwargs)
    data.plot(var2, ax=ax1, **common_kwargs)

    # ---------------------------------
    # Aesthetic cleanup
    # ---------------------------------
    for ax in [ax0, ax1]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_facecolor("white")

        # Clean, thin spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    # ---------------------------------
    # Colorbar
    # ---------------------------------
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=18)

    # ---------------------------------
    # Output
    # ---------------------------------
    if savefig is not None:
        plt.savefig(savefig, format="tiff", dpi=600, bbox_inches="tight")

    plt.show()

