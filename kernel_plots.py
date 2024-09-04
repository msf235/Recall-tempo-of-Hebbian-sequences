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

sns.set_palette('colorblind')

memory = mom.Memory('make_plots_cache')
# memory.clear()
memory_overlaps = mom.Memory('overlaps_cache')
# memory_overlaps.clear()
figdir=Path('plots')

legend=False
# legend=True
ext = '.pdf'

round_dig = 3
fig_pad = .01
figsize = (1.7, 1)
figsize2 = (1.5, 1)
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
    tau2 = [1, 1.1, 1.2, 1.3, 1.4, 1.7, 2]

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
    if w_params['wtype'] == 'linear':
        def w(t):
            b = (w_params['a'] <= t) * (t <= w_params['b'])
            return b * w_params['slope'] * (t-w_params['offset']) - w_params['mag1']
    elif w_params['wtype'] == 'double_exp':
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
    # tt = np.linspace(-.2,.2,1000)
    # ws = np.array([w(t) for t in tt])
    # plt.figure()
    # plt.plot(tt, ws)
    # plt.savefig('plots/w_fun.pdf')
    return w

def prep_system(w_params, inp_params, roundoff=1e-4):
    T_xi = inp_params['T_xi']
    P = inp_params['P']
    w = get_w(w_params)
    csvd = util.compute_coeffs(w, T_xi, P)
    csvd = {key: round(val, 4) for key, val in csvd.items() if
            (val != np.nan) & (abs(val) > roundoff)}
    return csvd

def mu_data(params, get_net=True):
    w_params = params['w_params']
    inp_params = params['inp_params']
    sim_params = params['sim_params']

    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    T_xi = inp_params['T_xi']
    P = inp_params['P']

    coeffs = prep_system(w_params, inp_params, roundoff=1e-4)
    gbar = 1/sum(coeffs.values())
    coeffs_norm = {key: val * gbar for key, val in coeffs.items()}
    ds = []
    ds += util.get_mf_linear_from_coeff(coeffs, params)[0]
    out = util.get_data_mf(coeffs, params)
    ds += out[0]
    # qs = out[1]
    # plt.figure()
    # plt.plot(tt, qs)
    # plt.savefig('plots/qs_plot.pdf')
    
    if get_net:
        out = util.get_data_network(coeffs, params, save_weights=False)
        ds += out[0]
        qs = out[1]
        # plt.figure()
        # plt.plot(tt, qs)
        # plt.savefig('plots/temp2.pdf')

    if 'tau' in sim_params:
        tau = sim_params['tau']
    else:
        tau = 1

    temp1 = w_params['mag2']*w_params['tau2'] \
                - w_params['mag1']*w_params['tau1']
    gbar2inv = T_xi * temp1
    gbar2 = 1/gbar2inv
    temp2 = w_params['mag1']*w_params['tau1']**2 \
                + w_params['mag2']*w_params['tau2']**2
    alpha1 = temp2 / gbar2inv
    temp = (1-np.exp(-T_xi/w_params['tau1']))*w_params['tau1']**2 \
            + (1-np.exp(-T_xi/w_params['tau2']))*w_params['tau2']**2
    alpha2 = temp / gbar2inv
    coeffs = {str(c): val for c, val in coeffs.items()}
    ds += [{'mu': k,
            'peak time': tau*(k-1)/alpha1 - alpha2/(2*alpha1**2),
            'peak diff': tau/alpha1,
            'mag': np.nan, 'type': 'approx', **coeffs} for k in range(1, P+1)]

    for d in ds:
        d['tau1'] = w_params['tau1']
        d['tau2'] = w_params['tau2']
        d['mag1'] = w_params['mag1']
        d['mag2'] = w_params['mag2']
        d['T_xi'] = T_xi

    return ds

def qu_data(params, get_net=True):
    w_params = params['w_params']
    inp_params = params['inp_params']
    sim_params = params['sim_params']

    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    T_xi = inp_params['T_xi']
    P = inp_params['P']

    coeffs = prep_system(w_params, inp_params)
    gbar = 1/sum(coeffs.values())
    coeffs_norm = {key: val * gbar for key, val in coeffs.items()}

    qs_lin = util.get_data_mf(coeffs, params)[1]
    if get_net:
        qs_net = util.get_data_network(coeffs, params, save_weights=False)[1]
        return {'linear': qs_lin, 'network': qs_net}

    return {'linear': qs_lin}


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
    ds, qsnet = util.get_data_network(A, params)
    dsn, qsmf = util.get_data_mf(A, params)
    ds += dsn
    return ds, qsnet

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

def make_df(params_list, mu_lim=None, run_num=None, get_net=True):
    ds = []
    if run_num is None:
        for i, params in enumerate(params_list):
            print("Run", i, '/', len(params_list)-1)
            ds += mu_data(params, get_net)
    elif run_num < len(params_list):
        print("Run", run_num, '/', len(params_list)-1)
        ds += mu_data(params_list[run_num], get_net)
    df = pd.DataFrame(ds)
    if 'mu' in df.columns and mu_lim is not None:
        df = df[df['mu']<=mu_lim]
    return df

def format_df(df: pd.DataFrame, inplace=False) -> pd.DataFrame | None:
    """Format dataframe for plotting."""
    replace_dict = {'tau1': r'$\tau_1$', 'tau2': r'$\tau_2$', 'mag1': r'$m_1$',
                    'mag2': r'$m_2$', 'peak diff': r'$d_{\mu}$', 'mu':
                    r'$\mu$', 'peak time': r'$t_{\mu}$', 'T_xi': r'$T_{\xi}$',
                    'mag': r'$p_{\mu}$'}
    return df.rename(columns=replace_dict, inplace=inplace)

# Fig 6B, 6C
def Txi_plots(params, run_num=None):
    params = copy.deepcopy(params)
    params_dsided_list = list_over_tau2([params], [.8, 1.2])
    params_dsided_list = list_over_mag1(params_dsided_list, [2, -.5])
    params_onesided = copy.deepcopy(params)
    params_onesided['w_params'].update({'mag1': 0})
    temp = params_dsided_list + [params_onesided]
    params_list = list_over_T_xi(temp, [.5, 1, 2, 3])
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
        putil.savefig(g, 'fig_6B') 

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
        putil.savefig(g, 'fig_6C') 

# Fig 6D
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
        putil.savefig(g, 'fig_6D_left') 

        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_yticks([0, 1, 2, 3])
        g = sns.lineplot(data=df, ax=ax, x=r'$\tau_1$', y=r'$d_{\mu}$',
                         hue=r'$\tau_2$',
                         hue_order=None, style='type',
                         style_order=['network', 'linear', 'approx'], alpha=0.7,
                        )
        putil.savefig(g, 'fig_6D_center') 


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
        putil.savefig(g, 'fig_6D_right') 

def format_file_str(fstr):
    """Format file string for use in plots. Replace '-' with 'm' and '.' with
    'p'."""
    fstr = fstr.replace('-', 'm')
    fstr = fstr.replace('.', 'p')
    return fstr

# Fig 7
def fast_and_slow_plots(params, run_num):
    params = copy.deepcopy(params)
    sim_params = params['sim_params']
    w_params = params['w_params']
    tau1 = w_params['tau1']
    mag1 = w_params['mag1']
    mag2 = w_params['mag2']

    if run_num is None or run_num == 0:
        fname_root = f'_changing_Txi_tau1_{tau1}_mag1_{mag1}_mag2_{mag2}'
        fname_root = format_file_str(fname_root)
        
        ds, qs = get_data_changing_Txi(params)
        tt = np.linspace(0, sim_params['T'], sim_params['t_steps'])

        fig, ax = plt.subplots(figsize=figsize2)
        putil.make_overlap_plot(qs[:900], tt[:900], peakyd=None, peakyu=None, ax=ax)
        peaks, maxs, peak_tidx = util.get_peaks(qs[:900], tt[:900])
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
        putil.savefig(ax, 'Fig_7D')

        df = pd.DataFrame(ds)
        df = df[df['mu']<=70]
        format_df(df, inplace=True)
        mus = r'$\mu$'
        peak_diff_str = r'$d_{\mu}$'
        fig, ax = plt.subplots(figsize=figsize2)
        g = sns.lineplot(data=df, ax=ax, x=mus, y=r'$t_{\mu}$',
                         hue=None, hue_order=None, style='type',
                         style_order=None)
        ymin, ymax = ax.get_ylim()
        ax.vlines([20, 30], ymin=ymin, ymax=ymax, colors='k',
                  linestyles='dashed')
        putil.savefig(ax, 'Fig_7B')

        fig, ax = plt.subplots(figsize=figsize2)
        g = sns.lineplot(data=df, ax=ax, x=mus, y=r'$d_{\mu}$',
                         hue=None, hue_order=None, style='type',
                         style_order=None)
        ymin, ymax = ax.get_ylim()

        dfavg = df.copy()
        dfavg['diff_avgs'] = np.nan
        idx = (3<=dfavg[mus])&(dfavg[mus]<18)
        dfavg['diff_avgs'][idx] = dfavg[idx][peak_diff_str].mean()
        sns.lineplot(ax=ax, data=dfavg, x=mus, y='diff_avgs',
                     linestyle='dotted', color='k',
                    )
        ymin, ymax = ax.get_ylim()
        dfavg['diff_avgs'] = np.nan
        idx = (20<=dfavg[mus])&(dfavg[mus]<30)
        dfavg['diff_avgs'][idx] = dfavg[idx][peak_diff_str].mean()
        sns.lineplot(ax=ax, data=dfavg, x=mus, y='diff_avgs',
                     linestyle='dotted', color='k',
                    )
        dfavg['diff_avgs'] = np.nan
        idx = 50<=dfavg[mus]
        dfavg['diff_avgs'][idx] = dfavg[idx][peak_diff_str].mean()
        sns.lineplot(ax=ax, data=dfavg, x=mus, y='diff_avgs',
                     linestyle='dotted', color='k',
                    )
        ymin, ymax = ax.get_ylim()
        ax.vlines([20, 30], ymin=ymin, ymax=ymax, colors='k',
                  linestyles='dashed')

        figname='diffs'+fname_root
        figpath = (figdir/figname).with_suffix(ext)
        putil.savefig(ax, 'Fig_7C')

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

def limiting_cases(base_params, T_xis, Ts, Pthr, Ns=None, run_num=1,
                    fignames=None, units=None):
    if fignames is None:
        fignames = ['mu_plot', 'tmu_plot', 'pmu_plot', 'inset', 'overlaps',
                    'overlaps_lin']
        fignames = ['kernel_' + f for f in fignames]
    params = copy.deepcopy(base_params)
    params_list = list_over_T_xi([params], T_xis)
    for k, paramd in enumerate(params_list):
        paramd['sim_params']['T'] = Ts[k]
        if Ns is not None:
            paramd['inp_params']['N'] = Ns[k]
    df = make_df(params_list, None, run_num, get_net=True)
    df = df.drop(columns=[k for k in df.columns if k.lstrip('-').isdigit()])
    df = df.drop(columns=['wtype', 'a', 'b', 'offset'])
    hue = r"$T_{\xi}$"
    tmu = r'$t_{\mu}$'
    dmu = r'$d_{\mu}$'
    mu = r'$\mu$'
    dfneti = df[df['type']=='network'].index
    dfmfi = df[df['type']=='mf'].index
    dflini = df[df['type']=='linear'].index
    format_df(df, inplace=True)
    style = 'type'
    baselinestr = r'$d_{\mu}$'
    integ_norm_str = r'$D$'

    if units is not None:
        df[hue] = df[hue]*1000
        df[hue] = df[hue].astype(int)
        df[tmu] = df[tmu]*1000
        df[dmu] = df[dmu]*1000
        hue = r'$T_{\xi}$ (ms)'
        tmu = r'$t_{\mu}$ (ms)'
        dmu = r'$d_{\mu}$ (ms)'
        df.rename(columns={r'$T_{\xi}$': hue, r'$d_{\mu}$': dmu,
                            r'$t_{\mu}$': tmu}, inplace=True)

        baselinestr = r'$d_{\mu}$ (ms)'
        integ_norm_str = r'$D$ (ms)'

    stylevals = df[style].unique()
    huevals = df[hue].unique()

    if len(df) == 0:
        return None

    df[hue] = df[hue].astype('category')

    dfc = df.copy()
    df.drop(df[(df[mu]>Pthr)].index, inplace=True)
    huevals = df[hue].unique()
    for hueval in huevals:
        dfhue_i = df[df[hue]==hueval].index
        ind = dfhue_i.intersection(dfneti)
        net_times = df.loc[ind][tmu]
        st_time = net_times.min()
        end_time = net_times.max()
        df_st_i = df[df[tmu]<st_time].index
        df_end_i = df[df[tmu]>end_time].index
        for stype in ['approx', 'linear', 'mf']:
            dft_i = df[df['type']==stype].index
            ind = dft_i.intersection(dfhue_i).intersection(df_st_i)
            df.drop(ind, inplace=True)
            ind = dft_i.intersection(dfhue_i).intersection(df_end_i)
            df.drop(ind, inplace=True)
    Txi_vals = df[hue].unique().to_numpy()
    Txi_vals = np.sort(Txi_vals)[::-1]

    # Fig S3e
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=df, ax=ax, x=mu, y=dmu,
                     hue=hue, style='type',
                     style_order=['network', 'mf', 'linear', 'approx'],
                     hue_order=Txi_vals,
                     alpha=0.7)
    ax.set_ylim([-.08, None])
    putil.savefig(ax, fignames[0], ncols=1)

    # Fig S3f
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=df, ax=ax, x=tmu, y=dmu,
                     hue=hue, style='type',
                     style_order=['network', 'mf', 'linear', 'approx'],
                     hue_order=Txi_vals,
                     alpha=0.7)
    ax.set_ylim([-.08, None])
    putil.savefig(ax, fignames[1], ncols=1)

    # Fig S3c/S3d
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=df, ax=ax, x=mu, y=r'$p_{\mu}$',
                     hue=hue, style='type',
                     style_order=['network', 'mf', 'linear'],
                     hue_order=Txi_vals,
                     alpha=0.7)
    ax.set_ylim([-.08, None])
    putil.savefig(ax, fignames[2], ncols=1)

    l_integ = []
    for huev in huevals:
        # for stylev in ['linear', 'network', 'mf']:
        for stylev in ['mf']:
            baseline = df[(df[hue]==huev)&(df[style]=='approx')][dmu].iloc[0]
            dfb = df[(df[hue]==huev)&(df[style]==stylev)]
            integral = ((dfb[dmu]-baseline)*dfb[dmu]).sum()
            l_integ.append({hue: huev, style: stylev,
                            'integral': np.abs(integral),
                            baselinestr: baseline})
            l_integ.append({hue: huev, style: 'unity',
                            'integral': baseline,
                            baselinestr: baseline})
    df_int = pd.DataFrame(l_integ)
    # integ_norm_str = 'Normalized deviation'
    df_int[integ_norm_str] = df_int['integral'] / df_int[baselinestr]
    # integ_y = 'integral'
    integ_y = integ_norm_str

    fig, ax = plt.subplots(figsize=(1.7*.45, 1*.45))
    g = sns.lineplot(data=df_int, ax=ax, x=baselinestr, y=integ_y,
                     style='type',
                     # style_order=['mf', 'unity'],
                     style_order=['mf'],
                     alpha=0.7)
    ax.set_ylim([-.08, None])
    putil.savefig(ax, fignames[3], ncols=1)

    plotdir_overlaps = Path(f'plots/{figname_base}_overlaps')
    plotdir_overlaps.mkdir(exist_ok=True)
    sim_params = params['sim_params']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    k1 = (len(params_list)//4) + 1
    k2 = min(len(params_list), 3)
    fig, axs = plt.subplots(k1, k2, figsize=(9,6))
    figlin, axslin = plt.subplots(k1, k2, figsize=(9,6))
    if k1 == 1 and k2 == 1:
        axs = [axs]
        axslin = [axslin]
    else:
        axs = axs.flatten()
        axslin = axslin.flatten()
    for k, param in enumerate(params_list):
        qsd = qu_data(param, get_net=True)
        ax = axs[k]
        overlaps = qsd['network'][:, 1:]
        ymax = np.max(overlaps)*1.3
        peakyu = ymax * .95
        peakyd = ymax * .9
        ax.set_ylabel(r'$q_{\mu}(t)$')
        ax.set_xlabel(r'$t$')
        putil.make_overlap_plot(overlaps, tt, peakyd, peakyu, ax)
        ax.set_ylim([None, ymax])
        ax.set_title(param['inp_params']['T_xi'])
        # putil.savefig(ax, f'{figname_base}_net_3_{k}')

        fign, axn = plt.subplots(figsize=figsize)
        if units is not None:
            ttn = tt*1000
        else:
            ttn = tt*1000
        putil.make_overlap_plot(overlaps, ttn, peakyd, peakyu, axn)
        axn.set_ylim([None, ymax])
        if units is not None:
            axn.set_xlabel(r'$t$ (ms)')
        else:
            axn.set_xlabel(r'$t$')
        axn.set_ylabel(r'$q_{\mu}(t)$')
        axn.set_title(param['inp_params']['T_xi'])
        # putil.savefig(axn, f'{figname_base}_overlaps/net_{k}',
                      # separate_legend=None)

        ax = axslin[k]
        overlaps = qsd['linear']
        ymax = np.max(overlaps)*1.3
        peakyu = ymax * .95
        peakyd = ymax * .9
        ax.set_ylabel(r'$q_{\mu}(t)$')
        ax.set_xlabel(r'$t$')
        putil.make_overlap_plot(overlaps, tt, peakyd, peakyu, ax)
        ax.set_ylim([None, ymax])
        ax.set_title(param['inp_params']['T_xi'])
        # putil.savefig(ax, f'{figname_base}_lin_3_{k}')

    fig.tight_layout()
    putil.savefig(fig, fignames[4], separate_legend=None)
    putil.savefig(figlin, fignames[5], separate_legend=None)
            
