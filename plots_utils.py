import pickle as pkl
import itertools
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import sim_utils as util

basic_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig_pad = .01
ext = '.pdf'
plotdir = Path('plots')
plotdir.mkdir(exist_ok=True)
(plotdir/'legend').mkdir(exist_ok=True)

rng = np.random.default_rng(13)
shuffle = rng.permutation(6)

overlap_cmap = np.array(sns.color_palette('deep', 6, as_cmap=True))[shuffle]

def make_overlap_plot(overlaps, tt, peakyd=None, peakyu=None, ax=None,
                      cmap=overlap_cmap):
    T = tt[-1]
    figs = []
    y_max = 0
    y_min = 1000
    peaks_list = []
    tick_spacing = 0.2
    peaks, maxs, peak_tidx = util.get_peaks(overlaps, tt)
    peaks = np.array(peaks)
    maxs = np.array(maxs)
    temp = np.where(peaks>=tt[-1])[0]
    if len(temp) > 0:
        fin_peak = temp[0] - 1
    else:
        fin_peak = -1
    peaks = peaks[:fin_peak]
    maxs = maxs[:fin_peak]
    peak_tidx = peak_tidx[:fin_peak]
    if len(peaks) < 10:
        maxs_avg = np.mean(maxs)
    else:
        maxs_avg = np.mean(maxs[5:10])
    idxs = maxs > maxs_avg*.2
    peaks = peaks[idxs]
    maxs = maxs[idxs]
    peak_tidx = np.array(peak_tidx)[idxs]
    idxs = np.diff(peaks) > 0
    peaks = np.concatenate([[peaks[0]], peaks[1:][idxs]])
    maxs = np.concatenate([[maxs[0]], maxs[1:][idxs]])
    peak_tidx = np.concatenate([[peak_tidx[0]], peak_tidx[1:][idxs]])
    peaks_list.append(peaks)
    k = max(peak_tidx[-1],3)

    y_max = max(np.max(overlaps[:k]).item(), y_max)
    y_min = min(np.min(overlaps[:k-2]).item(), y_min)
    if ax is None:
        fig, ax = plt.subplots(figsize=(2,1))
    else:
        fig = ax.figure
    K = overlaps.shape[1]
    cycle_len = 6
    # colors = cmap(np.arange(cycle_len)/ cycle_len)
    colors = cmap[:K]
    for k in range(K-1,-1,-1):
        ck = k % cycle_len
        ax.plot(tt, overlaps[:, k], linewidth=0.8, color=colors[ck])

    diff = y_max - y_min
    if peakyu is not None:
        peak_top = peakyu
    else:
        peak_top = y_max + .2*diff
    if peakyd is not None:
        peak_bottom = peakyd
    else:
        peak_bottom = y_max + .1*diff

    ax.vlines(peaks, peak_bottom, peak_top, colors=colors[:len(peaks)],
              linewidth=0.8)
    return fig

def get_coeff_order(df, coeff_key, sim_type):
    peaktimes = []
    coeff_vals = df[coeff_key].unique()
    for coeff_val in coeff_vals:
        filt = (df[coeff_key]==coeff_val)&(df['type']==sim_type)&(df['mu']<=5)
        temp = df[filt]['peak diff'].mean()
        peaktimes.append(temp)
    sortidx = np.argsort(peaktimes)[::-1]
    hue_order = coeff_vals[sortidx]
    return hue_order

def savefig(ax, fname_base, separate_legend=True, ncols=1, fig_pad=fig_pad):
    fig = ax.figure
    if separate_legend:
        ax.legend().remove()
    else:
        ax.legend(bbox_to_anchor=(1.05, 1.1))
    fname = (plotdir/f'{fname_base}').with_suffix(ext)
    fig.savefig(fname, bbox_inches='tight', pad_inches=fig_pad,
                transparent=True)
    figlegend = plt.figure()
    figlegend.legend(*ax.get_legend_handles_labels(), loc='center',
                     ncol=ncols)
    fnameleg = (plotdir/'legend'/f'{fname_base}_legend').with_suffix(ext)
    figlegend.savefig(fnameleg, bbox_inches='tight', pad_inches=fig_pad)
    plt.close("all")
    

