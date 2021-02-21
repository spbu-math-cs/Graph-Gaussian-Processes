import numpy as np
import osmnx
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_PEMS(G, vals, vertex_id, normalization, ax, fig, cax, vmin=None, vmax=None, filename=None, bbox=None,
              nodes_to_label=[], node_size=20, alpha=0.6, edge_linewidth=0.4,
              cmap_name='viridis', cut_colormap=False,
              plot_title=None):
    n, s, e, w = bbox  # bounds of crossroads
    mean, std = normalization
    vals = vals*std + mean
    vertex_id_dict = {vertex_id[i]: i for i in range(len(vertex_id))}

    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        if not cut_colormap:
            vmax = np.max(vals)
        else:
            vmax = np.sort(vals)[9*len(vals)//10]  # variance to high on distant points

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    colors = []
    for i in range(len(G)):
        if vertex_id_dict.get(i) is not None:
            val = vals[vertex_id_dict[i]]
            colors.append(cmap(norm(val)))
        else:
            colors.append((0, 0, 0, 1))  # black

    osmnx.plot_graph(G, show=False, close=False, bgcolor='w', node_color=colors, node_size=0,
                     edge_color='black', edge_linewidth=edge_linewidth, bbox=bbox, ax=ax)

    if plot_title is not None:
        ax.set_title(plot_title)

    nodes_to_label_set = set(nodes_to_label.ravel().tolist())  # nodes_to_label is nodes with data
    for node in G.nodes:
        if node not in nodes_to_label_set:
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            if s < y and y < n and w < x and x < e:  # select points at the crossroads
                val = vals[vertex_id_dict[node]]
                ax.scatter(x, y, s=node_size, color=cmap(norm(val)), alpha=alpha)

    for node in nodes_to_label_set:
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        if s < y and y <n and w < x and x < e:  # select points at the crossroads
            val = vals[vertex_id_dict[node]]
            ax.scatter(x, y, s=node_size, color=cmap(norm(val)), alpha=alpha)
            ax.scatter(x, y, s=3/20*node_size, color='white', alpha=alpha)

    # adding realworld map to the background
    ctx.add_basemap(ax=ax, crs='epsg:4326')
    ax.set_axis_off()
    if filename is not None:
        plt.savefig('plots/bay-traffic/{}.pdf'.format(filename), dpi=500)

    # fig, ax = plt.subplots(1, 1)

    if cut_colormap:
        cbar = fig.colorbar(cm.ScalarMappable(norm, cmap), orientation='vertical', extend='max', cax=cax)
        # cbar.ax.tick_params(labelsize=10)
    else:
        cbar = fig.colorbar(cm.ScalarMappable(norm, cmap), orientation='vertical', cax=cax)
        # cbar.ax.tick_params(labelsize=10)
    # ax.axis('off')
    if filename is not None:
        plt.savefig('plots/bay-traffic/сolorbars/colorabar_{}.pdf'.format(filename), dpi=500, transparent=True)
