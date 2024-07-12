print("running file.", flush=True)
import math
import sys
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import itertools
from pathlib import Path
import time
import model_output_manager_hash as mom
import kernels_params as ps
import sim_utils as util
import plots_utils as putil

memory = mom.Memory('make_plots_cache')
# memory.clear()
memory_overlaps = mom.Memory('overlaps_cache')
# memory_overlaps.clear()
figdir=Path('plots')

# legend=False
legend=True
ext = '.pdf'

round_dig = 3
fig_pad = .01
figsize = (1.7, 1)
figsize2 = (1.5, 1)
legend=False
plotdir = Path(f'plots')
plotdir.mkdir(exist_ok=True)

T_plot = 60

def make_tau_mag_combs(df):
    hue_var = r'$(\tau_2, m_1)$'
    df[hue_var] = ('(' + df['tau2'].astype(str) + ', ' + df['mag1'].astype(str)
                   + ')')
    return hue_var


def make_ps_kernel_kde():
    tau1 = [.2, .4, .8, 1.2, 1.6]
    # tau1 = [.2]
    tau2 = [1, 1.1, 1.2, 1.3, 1.4, 1.7, 2]
    # tau2 = [1]

    prd = itertools.product(tau1, tau2)
    prdf = []
    for p in prd:
        if sum(p) > 0.1:
            prdf.append(p)
            ps = [{'tau1': p[0], 'tau2': p[1]} for p in prdf]
    return ps

def make_ps_kernel_kde2():
    tau2 = [1, 1.1, 1.2, 1.3, 1.4, 1.7, 2]
    mag2 = [2, 4, 6, 10]
    # tau2 = [1]
    # mag2 = [10]

    prd = itertools.product(tau2, mag2)
    prdf = []
    for p in prd:
        if sum(p) > 0.1:
            prdf.append(p)
            ps = [{'tau2': p[0], 'mag2': p[1]} for p in prdf]
    return ps

def make_ps_kernel_heatmap():
    tau1 = [.2, .4, .6]
    tau2 = [1, 1.2, 1.4]

    prd = itertools.product(tau1, tau2)
    prdf = []
    for p in prd:
        if sum(p) > 0.1:
            prdf.append(p)
            ps = [{'tau1': p[0], 'tau2': p[1]} for p in prdf]
    return ps

def make_ps_kernel_heatmap_sensitivity(df1=.025, df2=.025):
    tau1 = [.2, .4, .6]
    tau2 = [1, 1.2, 1.4]

    dtau1 = [round(k - df1,3) for k in tau1] + [round(k + df1,3) for k in tau1]
    dtau2 = [round(k - df2,3) for k in tau2] + [round(k + df2,3) for k in tau2]

    prd1 = itertools.product(dtau1, tau2)
    prd2 = itertools.product(tau1, dtau2)
    prdf = []
    for p in prd1:
        if sum(p) > 0.1:
            prdf.append(p)
    for p in prd2:
        if sum(p) > 0.1:
            prdf.append(p)
    ps = [{'tau1': p[0], 'tau2': p[1]} for p in prdf]
    return ps

def make_ps_kernel_heatmap_sensitivity2(df1=.025, df2=.025):
    tau2 = [1, 1.2, 1.4]
    m2 = [5, 6, 7]

    dtau2 = [round(k - df1,3) for k in tau2] + [round(k + df1,3) for k in tau2]
    dm2 = [round(k - df2,3) for k in m2] + [round(k + df2,3) for k in m2]

    prd1 = itertools.product(dtau2, m2)
    prd2 = itertools.product(tau2, dm2)
    prdf = []
    for p in prd1:
        if sum(p) > 0.1:
            prdf.append(p)
    for p in prd2:
        if sum(p) > 0.1:
            prdf.append(p)
    ps = [{'tau2': p[0], 'mag2': p[1]} for p in prdf]
    return ps

def get_w(w_params):
    if w_params['type'] == 'linear':
        def w(t):
            b = (w_params['a'] <= t) * (t <= w_params['b'])
            return b * w_params['slope'] * (t-w_params['offset']) - w_params['mag1']
    elif w_params['type'] == 'double_exp':
        def w(t):
            b1 = (w_params['a'] <= t/w_params['tau1']) * (t < w_params['offset'])
            if b1 != 0:
                t1 = -b1 * w_params['mag1']*np.exp(t/w_params['tau1'])
            else:
                t1 = 0
            b2 = (w_params['offset'] < t) * (t/w_params['tau2'] <= w_params['b'])
            if b2 != 0:
                t2 = b2 * w_params['mag2']*np.exp(-t/w_params['tau2'])
            else:
                t2 = 0
            return t1 + t2
    return w

# @memory.cache
def prep_system(w_params, inp_params, roundoff=1e-4):
    T_xi = inp_params['T_xi']
    P = inp_params['P']
    w = get_w(w_params)
    csvd = util.compute_coeffs(w, T_xi, P)
    csvd = {key: round(val, 4) for key, val in csvd.items() if
            (val != np.nan) & (abs(val) > roundoff)}
    return csvd

def mu_data(params):
    w_params = params['w_params']
    inp_params = params['inp_params']
    sim_params = params['sim_params']

    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    T_xi = inp_params['T_xi']
    P = inp_params['P']

    coeffs = prep_system(w_params, inp_params)
    gbar = 1/sum(coeffs.values())
    coeffs_norm = {key: val * gbar for key, val in coeffs.items()}
    ds = []
    ds += util.get_data_mf_approx(coeffs, params)
    
    ds += util.get_data_network(coeffs, params, save_weights=False)
    # util.reset_inputs_and_weights()

    gbar2inv = T_xi*w_params['mag2']*w_params['tau2'] \
                - T_xi*w_params['mag1']*w_params['tau1']
    gbar2 = 1/gbar2inv
    temp = w_params['mag1']*w_params['tau1']**2 \
                + w_params['mag2']*w_params['tau2']**2
    alpha1 = temp / gbar2inv
    temp = (1-np.exp(-T_xi/w_params['tau1']))*w_params['tau1']**2 \
            + (1-np.exp(-T_xi/w_params['tau2']))*w_params['tau2']**2
    alpha2 = temp / gbar2inv
    coeffs = {str(c): val for c, val in coeffs.items()}
    ds += [{'mu': k,
            'peak time': (k-1)/alpha1 - alpha2/(2*alpha1**2),
            'peak diff': 1/alpha1,
            'mag': np.nan, 'type': 'approx', **coeffs} for k in range(1, P+1)]

    for d in ds:
        d['tau1'] = w_params['tau1']
        d['tau2'] = w_params['tau2']
        d['mag1'] = w_params['mag1']
        d['mag2'] = w_params['mag2']
        d['T_xi'] = T_xi
    df = pd.DataFrame(ds)
    dfc = df.drop(columns=[a for a in df.columns if a.lstrip('-').isdigit()])
    return ds


def get_data_changing_Txi(params, T_xis_type='speed_up_middle'):
    w_params = params['w_params']
    inp_params = params['inp_params']
    sim_params = params['sim_params']
    P = inp_params['P']

    T_xis = .6*np.ones(P)
    if T_xis_type == 'speed_up_middle':
        T_xis[20:30] = .3

    w = get_w(w_params)
    A = np.round(util.compute_A(w, T_xis), 4)
    qs = util.get_overlaps(A, params)[:, 1:]
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps'])
    peaks, peakmags, __ = util.get_peaks(qs, tt)
    peaks = np.array(peaks)
    diffs = np.nan * np.ones(peaks.shape)
    diffs[:-1] = np.diff(peaks)
    ds = []
    for k in range(len(peaks)):
        ds += [{'mu': k+1, 'peak time': peaks[k], 'peak diff': diffs[k],
                'mag': peakmags[k], 'type': 'network', 'tau1': w_params['tau1'],
                'tau2': w_params['tau2'], 'mag1': w_params['mag1'],
                'mag2': w_params['mag2'], 'T_xi_type': T_xis_type}]
    return ds, qs


def list_over_var(params_list, key1, key2, vs):
    ret_list = []
    for params in params_list:
        for v in vs:
            params = copy.deepcopy(params)
            params[key1].update({key2: v})
            ret_list.append(params)
    return ret_list

def list_over_T_xi(params_list, T_xis):
    return list_over_var(params_list, 'inp_params', 'T_xi', T_xis)

def list_over_tau2(params_list, tau2s):
    return list_over_var(params_list, 'w_params', 'tau2', tau2s)

def list_over_mag2(params_list, mag2s):
    return list_over_var(params_list, 'w_params', 'mag2', mag2s)

def list_over_mag1(params_list, mag1s):
    return list_over_var(params_list, 'w_params', 'mag1', mag1s)

def make_df(params_list, mu_lim=None, run_num=None):
    ds = []
    if run_num is None:
        for i, params in enumerate(params_list):
            print("Run", i, '/', len(params_list)-1)
            ds += mu_data(params)
    elif run_num < len(params_list):
        print("Run", run_num, '/', len(params_list)-1)
        ds += mu_data(params_list[run_num])
    df = pd.DataFrame(ds)
    if 'mu' in df.columns and mu_lim is not None:
        df = df[df['mu']<=mu_lim]
    return df


def format_df(df: pd.DataFrame, inplace=False) -> pd.DataFrame | None:
    """Format dataframe for plotting."""
    replace_dict = {'tau1': r'$\tau_1$', 'tau2': r'$\tau_2$', 'mag1': r'$m_1$',
                    'mag2': r'$m_2$', 'peak diff': r'$d_{\mu}$', 'mu':
                    r'$\mu$', 'peak time': r'$t_{\mu}$', 'T_xi': r'$T_{\xi}$',}
    return df.rename(columns=replace_dict, inplace=inplace)

# Fig 6b, 6c
def Txi_plots(params, run_num=None):
    params = copy.deepcopy(params)
    params_dsided_list = list_over_tau2([params], [.8, 1.2])
    # params_dsided_list = list_over_tau2([params], [.8])
    params_dsided_list = list_over_mag1(params_dsided_list, [-.5, 2])
    # params_dsided_list = list_over_mag1(params_dsided_list, [-2])
    params_onesided = copy.deepcopy(params)
    params_onesided['w_params'].update({'mag1': 0})
    temp = params_dsided_list + [params_onesided]
    params_list = list_over_T_xi(temp, [.5, 1, 2, 3])
    # params_list = list_over_T_xi(temp, [.5])
    df = make_df(params_list, 10, run_num)
    if len(df) > 0:
        hue = make_tau_mag_combs(df)
        hue_order = putil.get_coeff_order(df, hue, 'linear')
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize2)
        g = sns.lineplot(data=df, ax=ax, x=r'$T_{\xi}$', y=r'$d_{\mu}$',
                         hue=hue, hue_order=hue_order, style='type',
                         style_order=['network', 'linear', 'approx'], alpha=0.7,
                        )
        ax.set_ylim([0, 5.5])
        ax.set_yticks(np.arange(0, 6, 1))
        putil.savefig(g, 'fig_6b') 

    params = copy.deepcopy(params)
    params_dsided_list = list_over_mag2([params], [2, 4])
    params_list = list_over_T_xi(params_dsided_list, [.5, 1, 2, 3])
    df = make_df(params_list, 10, run_num)
    if len(df) > 0:
        format_df(df, inplace=True)
        df[r'$m_2$'] = df[r'$m_2$'].astype('category')
        fig, ax = plt.subplots(figsize=figsize2)
        g = sns.lineplot(data=df, ax=ax, x=r'$T_{\xi}$', y=r'$d_{\mu}$',
                         hue=r'$m_2$', hue_order=None, style='type',
                         style_order=['network', 'linear', 'approx'], alpha=0.7,
                        )
        ax.set_ylim([0, 3.6])
        ax.set_yticks([0, 1, 2, 3])
        putil.savefig(g, 'fig_6c') 

# Fig 6d
def params_combos_plots(params, run_num):
    params = copy.deepcopy(params)
    ps = make_ps_kernel_kde()
    params_list = []
    ylims = [0.5, 3.3]
    for p in ps:
        d = copy.deepcopy(params)
        d['w_params']['tau1'] = p['tau1']
        d['w_params']['tau2'] = p['tau2']
        params_list.append(d)
    df = make_df(params_list, 10, run_num)
    # dfc = df.drop(columns = [str(k) for k in range(-6,9)])
    if len(df)> 0:
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_yticks([0, 1, 2, 3])
        g = sns.lineplot(data=df, ax=ax, x=r'$\tau_2$', y=r'$d_{\mu}$',
                         hue=r'$\tau_1$',
                         hue_order=None, style='type',
                         style_order=['network', 'linear', 'approx'], alpha=0.7,
                        )
        putil.savefig(g, 'fig_6d_left') 

        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_yticks([0, 1, 2, 3])
        g = sns.lineplot(data=df, ax=ax, x=r'$\tau_1$', y=r'$d_{\mu}$',
                         hue=r'$\tau_2$',
                         hue_order=None, style='type',
                         style_order=['network', 'linear', 'approx'], alpha=0.7,
                        )
        putil.savefig(g, 'fig_6d_center') 


    params = copy.deepcopy(params)
    ps = make_ps_kernel_kde2()
    params_list = []
    for p in ps:
        d = copy.deepcopy(params)
        d['w_params']['tau2'] = p['tau2']
        d['w_params']['mag2'] = p['mag2']
        params_list.append(d)
    df = make_df(params_list, 10, run_num)
    dfn = df.copy()
    if len(df)> 0:
        dfn = dfn.drop(columns=[r'T_xi', 'h_sig', 'tau1'])
        dfn2 = dfn[dfn['tau2']==1.0]
        dfn2[dfn2['mag2']==10]
        format_df(df, inplace=True)
        df['$m_2$'] = df['$m_2$'].astype('category')
        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_yticks([0, 1, 2, 3])
        g = sns.lineplot(data=df, ax=ax, x=r'$m_2$', y=r'$d_{\mu}$',
                         hue=r'$\tau_2$',
                         hue_order=None, style='type',
                         style_order=['network', 'linear', 'approx'],
                         alpha=0.7,
                        )
        putil.savefig(g, 'fig_6d_right') 

def format_file_str(fstr):
    """Format file string for use in plots. Replace '-' with 'm' and '.' with
    'p'."""
    fstr = fstr.replace('-', 'm')
    fstr = fstr.replace('.', 'p')
    return fstr

# Fig 7
def fast_and_slow_plots(base_params, run_num):
    params = copy.deepcopy(base_params)
    sim_params = params['sim_params']
    w_params = params['w_params']

    if run_num is None or run_num == 0:
        tau1 = 0.25
        mag1 = 2
        mag2 = 2
        w_params.update(dict(tau1=tau1, mag1=mag1, mag2=mag2))
        fname_root = f'_changing_Txi_tau1_{tau1}_mag1_{mag1}_mag2_{mag2}'
        fname_root = format_file_str(fname_root)
        
        ds, qs = get_data_changing_Txi(params)
        tt = np.linspace(0, sim_params['T'], sim_params['t_steps'])

        fig, ax = plt.subplots(figsize=figsize2)
        putil.make_overlap_plot(qs[:300], tt[:300], peakyd=None, peakyu=None, ax=ax)
        peaks, maxs, peak_tidx = util.get_peaks(qs[:300], tt[:300])
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
        ax.set_ylabel(r'$q_{\mu}(t)$')
        ax.set_xlabel(r'$t$')
        ymin, ymax = ax.get_ylim()
        ax.vlines([peaks[19], peaks[29]], ymin=ymin, ymax=ymax, colors='k',
                  linestyles='dashed')
        putil.savefig(ax, 'Fig_7d')

        df = pd.DataFrame(ds)
        df = df[df['mu']<=70]
        format_df(df, inplace=True)
        mus = r'$\mu$'
        peak_diff_str = r'$d_{\mu}$'
        fig, ax = plt.subplots(figsize=figsize2)
        g = sns.lineplot(data=df, ax=ax, x=mus, y=r'$t_{\mu}$',
                         hue=None, hue_order=None, style=None,
                         style_order=None)
        ymin, ymax = ax.get_ylim()
        ax.vlines([20, 30], ymin=ymin, ymax=ymax, colors='k',
                  linestyles='dashed')
        putil.savefig(ax, 'Fig_7b')

        fig, ax = plt.subplots(figsize=figsize2)
        g = sns.lineplot(data=df, ax=ax, x=mus, y=r'$d_{\mu}$',
                         hue=None, hue_order=None, style=None,
                         style_order=None)
        ymin, ymax = ax.get_ylim()

        dfavg = df.copy()
        dfavg['diff_avgs'] = np.nan
        idx = (3<=dfavg[mus])&(dfavg[mus]<18)
        dfavg['diff_avgs'][idx] = dfavg[idx][peak_diff_str].mean()
        sns.lineplot(ax=ax, data=dfavg, x=mus, y='diff_avgs',
                     linestyle='dotted', color='k'
                    )
        ymin, ymax = ax.get_ylim()
        dfavg['diff_avgs'] = np.nan
        idx = (20<=dfavg[mus])&(dfavg[mus]<30)
        dfavg['diff_avgs'][idx] = dfavg[idx][peak_diff_str].mean()
        sns.lineplot(ax=ax, data=dfavg, x=mus, y='diff_avgs',
                     linestyle='dotted', color='k'
                    )
        dfavg['diff_avgs'] = np.nan
        idx = 50<=dfavg[mus]
        dfavg['diff_avgs'][idx] = dfavg[idx][peak_diff_str].mean()
        sns.lineplot(ax=ax, data=dfavg, x=mus, y='diff_avgs',
                     linestyle='dotted', color='k'
                    )
        ymin, ymax = ax.get_ylim()
        ax.vlines([20, 30], ymin=ymin, ymax=ymax, colors='k',
                  linestyles='dashed')

        figname='diffs'+fname_root
        figpath = (figdir/figname).with_suffix(ext)
        putil.savefig(ax, 'Fig_7c')

        if not legend:
            g.legend_ = None
        else:
            ax.legend(bbox_to_anchor=(1.1, 1.05), loc=2, borderaxespad=0.)

        figlegend = plt.figure(figsize=(1,1))
        figlegend.legend(*ax.get_legend_handles_labels(), loc='center',
                         ncol=1)
        fnameleg = (figdir/'legend'/figname).with_suffix(ext)
        figlegend.savefig(fnameleg, bbox_inches='tight', transparent=True,
                          pad_inches=.01)
