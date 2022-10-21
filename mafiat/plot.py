#!/usr/bin/env python
"""Functions supporting the plotting of twist results."""
import matplotlib.pyplot as plt

def plot_2d_map(map, crds, cmap='bwr', clim=None, xlim=None, ylim=None, clabel=None, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    if clim is not None:
        p = ax.pcolormesh(crds[0]/1e6, crds[1]/1e6, map.T, vmin=clim[0], vmax=clim[1], cmap=plt.get_cmap(cmap))
    else:
        p = ax.pcolormesh(crds[0]/1e6, crds[1]/1e6, map.T, cmap=plt.get_cmap(cmap))

    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    cbar = plt.gcf().colorbar(p, ax=ax, pad=0.01)
    if clabel is not None:
        cbar.set_label(clabel)
    ax.set_xlabel('X Position [Mm]')
    ax.set_ylabel('Y Position [Mm]')
