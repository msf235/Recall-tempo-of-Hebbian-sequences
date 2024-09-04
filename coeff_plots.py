import sim_utils as util
import plots_utils as putil
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import seaborn as sns
import copy
import argparse
import sys

round_dig = 3
machine_eps = 1e-14
fig_pad = .01
ncols = 6
T_plot = 60
figsize = (1.7, 1)
figsize1 = (1.7, 1)
figsize2 = (1.4, 1)
legend = False
plotdir = Path('plots')
plotdir.mkdir(exist_ok=True)
(plotdir/'legend').mkdir(exist_ok=True)
ext = '.pdf'


def make_coeffs_one_forward():

    a0s = [-.4, 0, .4]
    a1s = [.6, .8, 1]

    prd = itertools.product(a0s, a1s)
    ps = [{0: round(p[0], 3), 1: round(p[1], 3)} for p in prd]
    ps += [{0: 0, 1: .1}]
    # ps.pop(7)
    # ps2 = [{0: small_p[0], 1: small_p[1]}]
    # ps += ps2
    # ps2 += [{0: 0, 1: 1}]
    return ps

def make_coeffs_one_forward_const_g(csum=1.5):
    a0s = np.array([-10, -4, -3, -2, -1])
    a1s = csum - a0s
    ps = [{0: a0s[k], 1: a1s[k]} for k in range(len(a0s))]

    return ps

def make_coeffs_one_forward_heatmap(threshold=0):

    a0s = [-.4, -.2, 0, .2, .4]
    a1s = [.2, .4, .6, .8]
    a1s = [.4, .6, .8, 1]

    prd = itertools.product(a0s, a1s)
    prd2 = []
    for p in prd:
        if sum(p) > threshold:
            prd2.append(p)
    ps = [{0: round(p[0], 3), 1: round(p[1], 3)} for p in prd2]
    return ps


def make_coeffs_ofob():
    a_m1s = np.linspace(-.2, .2, 3)
    # a_m1s = np.array([-.4, -.2])
    # a_m1s = np.array([-.2])
    a_0s = np.linspace(-.2, .2, 2)
    a_1s = np.linspace(.8, 1., 2)
    prd = np.round(list(itertools.product(a_m1s, a_0s, a_1s)), 2)
    ps = [{-1: p[0], 0: p[1], 1: p[2]} for p in prd]
    ps.pop(2) 
    return ps

def make_coeffs_ofob_heatmap(threshold=0.1, bar=False):

    ps = []
    if bar:
        am1bs = [-.4, -.2, 0, .2]
        a0bs = [-.2, 0, .2]
        a1s = [.4, .6, .8]
        prd = itertools.product(am1bs, a0bs, a1bs)
        for p in prd:
            csum = sum(p)
            if csum > threshold:
                a0 = round(p[1] / csum, 3)
                am1 = round(p[0] / csum, 3)
                a1 = round(p[2] / csum, 3)
                ps += [{-1: round(am1, 3), 0: round(a0, 3), 1: round(a1, 3)}]
    else:
        # am1s = [-.2]
        am1s = [-.2, .2]
        a0s = [-.4, -.2, 0, .2, .4]
        # a1s = [.2, .4, .6, .8]
        a1s = [.4, .6, .8, 1]

    prd = itertools.product(am1s, a0s, a1s)
    prd2 = []
    for p in prd:
        if sum(p) > threshold:
            prd2.append(p)
    ps = [{-1: p[0], 0: p[1], 1: p[2]} for p in prd2]
    return ps

def make_coeffs_two_forward(threshold=0.1):
    a_0s = [-.4, 0, .4]
    a_1s = [.6, 1.0]
    a_2s = [0, .4]
    prd = np.round(list(itertools.product(a_0s, a_1s, a_2s)), 2)
    prd2 = []
    for p in prd:
        if sum(p) > threshold:
            prd2.append(p)
    ps = [{0: p[0], 1: p[1], 2: p[2]} for p in prd2]
    ps.append({0: 0.4, 1: 0.6, 2: -0.1})
    return ps

def make_coeffs_two_forward_heatmap(threshold=0.1):
    a0s = [-.4, -.2, 0, .2, .4]
    a1s = [.2, .4, .6, .8]
    a1s = [.4, .6, .8, 1]
    a2s = [.2]
    prd = np.round(list(itertools.product(a0s, a1s, a2s)), 2)
    prd2 = []
    for p in prd:
        if sum(p) > threshold:
            prd2.append(p)
    ps = [{0: p[0], 1: p[1], 2: p[2]} for p in prd2]
    return ps


def make_coeff_combs(df, normalize=False):
    if len(df) == 0:
        return None
    coeffs_keys = [str(col) for col in df.columns
                   if isinstance(col, int) or col.lstrip('-').isdigit()]
    cn = '('
    hue = r'$('
    csum = 0
    for i0, k in enumerate(coeffs_keys):
        csum += df[k]
    for i0, k in enumerate(coeffs_keys):
        if normalize:
            cn += round(df[k]/csum, 3).astype(str)
            hue += f'\\bar{{a}}_{{{k}}}'
        else:
            cn += df[k].astype(str)
            hue += f'a_{{{k}}}'
        if i0 < len(coeffs_keys)-1:
            cn += ', '
            hue += ', '
    cn += ')'
    hue += ')$'
    df[hue] = cn
    return hue

def stability_noise_data(h_sigs, a0s, params, run_num):
    sim_params = params['sim_params']
    inp_params = params['inp_params']
    theta = sim_params['theta']
    sig = sim_params['sig']
    rspan = sim_params['rspan']
    gcrits = util.G(np.array(h_sigs)**2, theta, sig, rspan)
    offsets = [-.05, -0.02, -.01, -.005, 0, .01, 0.02, .05] 
    csumcrits = 1/gcrits
    arg_list = []
    ds = []
    for a0 in a0s:
        for k, c in enumerate(csumcrits):
            h_sig = h_sigs[k]
            for o in offsets:
                pc = copy.deepcopy(params)
                pc['sim_params']['h_sig'] = h_sig
                a1 = round(c - a0 + o, 3)
                coeffs = {0: a0, 1: a1}
                arg_list.append((coeffs, pc))

    if run_num is not None: 
        if run_num > 0 and run_num <= len(arg_list):
            print("Run", run_num, '/', len(arg_list))
            ds = util.get_data_network(*arg_list[run_num-1])
        else:
            ds = []
    else:
        ds = []
        for k, (coeffs, pc) in enumerate(arg_list):
            print("Run", k+1, '/', len(arg_list))
            ds += util.get_data_network(coeffs, pc)[0]
    return ds

def mu_data(coeffs, params, h_sigs=None, save_weights=True,
            mean_field=True): # Hacky add-on, TODO: fix

    sim_params = params['sim_params']
    P = params['inp_params']['P']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)

    if h_sigs is None:
        h_sigs = [sim_params['h_sig']]
    ds = []
    for h_sig in h_sigs:
        params_h = copy.deepcopy(params)
        params_h['sim_params']['h_sig'] = h_sigs[0]
        ds += util.get_data_network(coeffs, params_h,
                                    save_weights=save_weights)[0]
    ds += util.get_mf_linear_from_coeff(coeffs, params_h)[0]
    if mean_field:
        ds += util.get_data_mf(coeffs, params_h)[0]
    df = pd.DataFrame(ds)

    alpha1, alpha2 = util.get_alphas(coeffs)
    coeffs = {str(k): v for k, v in coeffs.items()}
    abar = np.sum(list(coeffs.values()))
    coeffsn = {k: v/abar for k, v in coeffs.items()}

    ds += [{'mu': k,
            'peak time': k/alpha1 - alpha2/(2*alpha1**2),
            # 'peak time': (k+1)/alpha1 - alpha2/alpha1**2,
            'peak diff': 1/alpha1,
            'mag': np.nan,
            'type': 'approx',
            **coeffs}
            for k in range(1, P+1)]
    return ds

def time_data(coeffs, params, save_weights=True):
    sim_params = params['sim_params']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    ds = []
    qss = {}
    out = util.get_gfun_network_data(coeffs, params)
    ds += out[0]
    qss['network'] = util.get_data_network(coeffs, params, False)
    ds += util.get_gfun_mf_data(coeffs, params)[0]
    # qss['linear'] = util.get_linear_from_coeff(coeffs, params)
    qss['linear'] = util.get_mf_linear_from_coeff(coeffs, params)[1]
    gbar = 1/sum(coeffs.values())
    coeffs = {str(key): val for key, val in coeffs.items()}
    ds += [{'t': t, 'g(t)': gbar, 'Garg': None, 'type': 'approx', **coeffs} for
           t in tt]
    return ds, qss


def format_df(df: pd.DataFrame, inplace=False) -> pd.DataFrame | None:
    """Format dataframe for plotting."""
    replace_dict = {'0': r'$a_0$', '1': r'$a_1$', '2': r'$a_2$', '-1':
                    r'$a_{-1}$', 'peak diff': r'$d_{\mu}$', 'mu': r'$\mu$',
                    'peak time': r'$t_{\mu}$', 't': r'$t$', 'g(t)': r'$g(t)$',
                    'csum': r'$a_0+a_1$', 'mag': r'$p_{\mu}$'}
    return df.rename(columns=replace_dict, inplace=inplace)

def make_df(data_fun, coeff_list, params, mu_lim=None, run_num=None):
    if run_num is None:
        ds = []
        for i, coeffs in enumerate(coeff_list):
            print("Run", i+1, '/', len(coeff_list))
            ds += data_fun(coeffs, params)
    else:
        if run_num > 0 and run_num <= len(coeff_list):
            print("Run", run_num, '/', len(coeff_list))
            ds = data_fun(coeff_list[run_num-1], params)
        else:
            ds = []
    df = pd.DataFrame(ds)
    if 'peak time' in df.columns:
        df = df[df['peak time']<=T_plot]
    if 'mu' in df.columns and mu_lim is not None:
        df = df[df['mu']<=mu_lim]
    return df

def make_df_h_sigs(a0s, h_sigs, params, mu, run_num=None):
    ds = stability_noise_data(h_sigs, a0s, params, run_num)
    df = pd.DataFrame(ds)
    if len(df) == 0:
        return df
    df = df[df['mu']==mu]
    df['csum'] = (df['0'] + df['1']).round(3)
    return df

# Fig 1
def model_overview_plots(params, figsize=figsize2,
                         cmap=putil.overlap_cmap, run_num=None):
    if run_num is not None and run_num != 0:
        return -1
    params = copy.deepcopy(params)
    sim_params = params['sim_params']
    T = 30
    sim_params['T'] = T
    params['inp_params']['P'] = 40
    t_steps = sim_params['t_steps']

    coeffs = {0: 0, 1: 1}
    tt = np.linspace(0, T, t_steps+1)
    r_soln = util.simulate_rnn_subset(coeffs, params, 10)

    # Fig 1c
    fig, ax = plt.subplots(figsize=figsize)
    K = r_soln.shape[1]
    cycle_len = 6
    colors = cmap[:K]
    for k in range(K-1,-1,-1):
        ck = k % cycle_len
        ax.plot(tt, r_soln[:, k], linewidth=0.8, color=colors[ck])
    ax.set_ylim([-1.2, 1.2])
    ax.set_ylabel(r'$r_k(t)$')
    ax.set_xlabel(r'$t$')
    putil.savefig(ax, 'fig_1c')

    overlaps = util.get_overlaps(coeffs, params)[:, 1:]
    ymax = np.max(overlaps)*1.3
    peakyu = ymax * .95
    peakyd = ymax * .9

    # Fig 1d
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')

    putil.make_overlap_plot(overlaps, tt, peakyd, peakyu, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_1d')


# Fig 3
def one_forward_plots(base_params, run_num=None):
    params = copy.deepcopy(base_params)
    coeff_list = make_coeffs_one_forward()

    # Fig 3b
    df = make_df(mu_data, coeff_list, params, 70, run_num)
    if len(df) > 0:
        hue = make_coeff_combs(df)
        hue_order = putil.get_coeff_order(df, hue, 'linear')
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize)
        g = sns.lineplot(data=df, ax=ax, x=r'$\mu$', y=r'$t_{\mu}$',
                         hue=hue, hue_order=hue_order, style='type',
                         style_order=['network', 'linear'], alpha=0.7)
        putil.savefig(ax, 'fig_3b', ncols=7) 

    # Fig 3c
    data_fun = lambda coeff, params: time_data(coeff, params, False)[0]
    df = make_df(data_fun, coeff_list, params, None, run_num)
    if len(df) > 0:
        hue = make_coeff_combs(df)
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_ylim(ylims)
        g = sns.lineplot(data=df, ax=ax, x=r'$t$', y=r'$g(t)$',
                         hue=hue, hue_order=hue_order, style='type',
                         style_order=['network', 'approx'], alpha=0.7)
        ax.axhline(20 / np.sqrt(2*np.pi), color='k', linestyle='dotted')
        ax.set_ylim([0, None])
        putil.savefig(ax, 'fig_3c', ncols=7) 

    # Fig 3d
    ylims = [0, 3.2]
    params = copy.deepcopy(base_params)
    sim_params = params['sim_params']
    coeff_list = make_coeffs_one_forward_heatmap()
    mu = 70
    df = make_df(mu_data, coeff_list, params, mu, run_num)
    if len(df) > 0:
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_xticks([-.4, 0, .4])
        g = sns.lineplot(data=df, ax=ax, x=r'$a_0$', y=r'$d_{\mu}$',
                         hue=r'$a_1$', style='type',
                         style_order=['network', 'linear'], alpha=0.7)
        putil.savefig(ax, 'fig_3d_left', ncols=1) 

    # Fig 3d
    if len(df) > 0:
        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        g = sns.lineplot(data=df, ax=ax, x=r'$a_1$', y=r'$d_{\mu}$',
                         hue=r"$a_0$", hue_order=None, style='type',
                         style_order=['network', 'linear'], alpha=0.7)
        putil.savefig(ax, 'fig_3d_right', ncols=1) 

    # Fig 3e
    params = copy.deepcopy(base_params)
    params['sim_params']['T'] = 200
    params['sim_params']['t_steps'] = 3000
    sim_params = params['sim_params']
    dtype = ['network', 'mf_approx']
    mu = 70
    df = make_df_h_sigs([-.2, 0, .1], [0, 0.1], params, mu, run_num)
    if len(df) > 0:
        df['h_sig'] = df['h_sig'].astype('category')
        h_sigs = df['h_sig'].unique()
        ymax = df['mag'].max()
        ystr = f'$p_{{{mu}}}$'
        gcrits = util.G(np.array(h_sigs)**2, sim_params['theta'],
                        sim_params['sig'], sim_params['rspan'])
        format_df(df, inplace=True)
        df.rename(columns={r'$p_{\mu}$': ystr}, inplace=True)
        fig, ax = plt.subplots(figsize=figsize2)
        g = sns.lineplot(data=df, ax=ax, x=r'$a_0+a_1$', y=ystr, hue='h_sig')
        ax.vlines(1/gcrits, 0, ymax, linestyles='--',
                  colors=[g.lines[0].get_color(), g.lines[1].get_color()])
        putil.savefig(ax, 'fig_3e', ncols=1) 


    if run_num is not None and run_num != 0:
        return 0

    params = copy.deepcopy(base_params)
    sim_params = params['sim_params']
    T = sim_params['T']
    t_steps = sim_params['t_steps']
    tt = np.linspace(0, T, t_steps+1)
    dt = tt[1]-tt[0]
    tk = int(30/dt)
    tt = tt[:tk]

    # Fig 3A
    coeff_list = make_coeffs_one_forward()
    coeffs = {0: 0.4, 1: 0.6}
    overlaps = util.get_overlaps(coeffs, params)[:tk, 1:]
    ymax = np.max(overlaps)*1.3
    fig, ax = plt.subplots(figsize=figsize2)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')
    putil.make_overlap_plot(overlaps, tt, ymax*.9, ymax*.95, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_3A')

    # Fig 3B
    coeffs = {0: -0.4, 1: 0.6}
    overlaps = util.get_overlaps(coeffs, params)[:tk, 1:]
    ymax = np.max(overlaps)*1.3
    fig, ax = plt.subplots(figsize=figsize2)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')
    putil.make_overlap_plot(overlaps, tt, ymax*.9, ymax*.95, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_3B')

    # Fig 3C
    coeffs = {0: 0, 1: 0.1}
    overlaps = util.get_overlaps(coeffs, params)[:tk, 1:]
    ymax = np.max(overlaps)*1.3
    fig, ax = plt.subplots(figsize=figsize2)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')
    putil.make_overlap_plot(overlaps, tt, ymax*.9, ymax*.95, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_3C')


# Fig 4
def ofob_plots(base_params, run_num=None):
    params = copy.deepcopy(base_params)
    coeff_list = make_coeffs_ofob()
    # coeff_list = make_coeffs_ofob_heatmap()
    df = make_df(mu_data, coeff_list, params, 70, run_num)
    style_order=['network', 'linear', 'approx']

    # Fig 4b
    if len(df) > 0:
        hue = make_coeff_combs(df)
        hue_order = putil.get_coeff_order(df, hue, 'linear')
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize)
        g = sns.lineplot(data=df, ax=ax, x=r'$\mu$', y=r'$t_{\mu}$',
                         hue=hue, hue_order=hue_order, style='type',
                         style_order=style_order,
                         alpha=0.7,)
        putil.savefig(ax, 'fig_4b', ncols=2) 

    coeff_list = make_coeffs_ofob_heatmap()
    df = make_df(mu_data, coeff_list, params, 70, run_num)
    dfp = df.copy()
    # Fig 4c, 4d
    if len(df) > 0:
        ylims = [0, 3.2]
        df = df[df['-1']==-.2]
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_xticks([-.4, 0, .4])
        g = sns.lineplot(data=df, ax=ax, x=r'$a_0$', y=r'$d_{\mu}$',
                         hue=r'$a_1$', hue_order=None, style='type',
                         style_order=style_order,
                         alpha=0.7,)
        putil.savefig(ax, 'fig_4c_left', ncols=1) 

        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        g = sns.lineplot(data=df, ax=ax, x=r'$a_1$', y=r'$d_{\mu}$',
                         hue=r'$a_0$', hue_order=None, style='type',
                         style_order=style_order,
                         alpha=0.7,)
        putil.savefig(ax, 'fig_4c_right', ncols=1) 

        df = dfp[dfp['-1']==.2]
        ylims = [0, 8]
        # ylims = None
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_xticks([-.4, 0, .4])
        g = sns.lineplot(data=df, ax=ax, x=r'$a_0$', y=r'$d_{\mu}$',
                         hue=r'$a_1$', hue_order=None, style='type',
                         style_order=style_order,
                         alpha=0.7,)
        putil.savefig(ax, 'fig_4d_left', ncols=1) 

        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        g = sns.lineplot(data=df, ax=ax, x=r'$a_1$', y=r'$d_{\mu}$',
                         hue=r'$a_0$', hue_order=None, style='type',
                         style_order=style_order,
                         alpha=0.7,)
        putil.savefig(ax, 'fig_4d_right', ncols=1) 

    ## Popout overlap plots
    if run_num is not None and run_num != 0:
        return 0

    params = copy.deepcopy(base_params)
    sim_params = params['sim_params']
    T = sim_params['T']
    t_steps = sim_params['t_steps']
    tt = np.linspace(0, T, t_steps+1)
    dt = tt[1]-tt[0]
    tk = int(30/dt)
    tt = tt[:tk]

    # Fig 4A
    coeffs = {-1: 0.2, 0: 0.2, 1: 0.8}
    overlaps = util.get_overlaps(coeffs, params)[:tk, 1:]
    ymax = np.max(overlaps)*1.3
    fig, ax = plt.subplots(figsize=figsize2)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')
    putil.make_overlap_plot(overlaps, tt, ymax*.9, ymax*.95, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_4A')

    # Fig 4B
    coeffs = {-1: -0.2, 0: -0.2, 1: 0.8}
    overlaps = util.get_overlaps(coeffs, params)[:tk, 1:]
    ymax = np.max(overlaps)*1.3
    fig, ax = plt.subplots(figsize=figsize2)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')
    putil.make_overlap_plot(overlaps, tt, ymax*.9, ymax*.95, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_4B')

    plt.close("all")

# Fig 5
def two_forward_plots(base_params, run_num=None):
    params = copy.deepcopy(base_params)
    coeff_list = make_coeffs_two_forward()
    df = make_df(mu_data, coeff_list, params, 70, run_num)

    # Fig 5b
    if len(df) > 0:
        hue = make_coeff_combs(df)
        hue_order = putil.get_coeff_order(df, hue, 'linear')
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize)
        g = sns.lineplot(data=df, ax=ax, x=r'$\mu$', y=r'$t_{\mu}$',
                         hue=hue, hue_order=hue_order, style='type',
                         style_order=['network', 'linear', 'approx'], alpha=0.7,)
        putil.savefig(ax, 'fig_5b', ncols=2) 

    coeff_list = make_coeffs_two_forward_heatmap()
    df = make_df(mu_data, coeff_list, params, 70, run_num)
    if len(df) > 0:
        ylims = [0, 3.2]
        # df = df[df['2']==.2]
        format_df(df, inplace=True)
        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        ax.set_xticks([-.4, 0, .4])
        g = sns.lineplot(data=df, ax=ax, x=r'$a_0$', y=r'$d_{\mu}$',
                         hue=r'$a_1$', hue_order=None, style='type',
                         style_order=['network', 'linear', 'approx'],
                         alpha=0.7,)
        putil.savefig(ax, 'fig_5C_left', ncols=1) 

        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylim(ylims)
        g = sns.lineplot(data=df, ax=ax, x=r'$a_1$', y=r'$d_{\mu}$',
                         hue=r'$a_0$', hue_order=None, style='type',
                         style_order=['network', 'linear', 'approx'],
                         alpha=0.7,)
        putil.savefig(ax, 'fig_5C_right', ncols=1) 


    ## Popout overlap plots
    if run_num is not None and run_num != 0:
        return 0

    params = copy.deepcopy(base_params)
    sim_params = params['sim_params']
    T = sim_params['T']
    t_steps = sim_params['t_steps']
    tt = np.linspace(0, T, t_steps+1)
    dt = tt[1]-tt[0]
    tk = int(30/dt)
    tt = tt[:tk]

    # Fig 5D
    coeffs = {0: 0.4, 1: 0.6, 2: -0.1}
    overlaps = util.get_overlaps(coeffs, params)[:tk, 1:]
    ymax = np.max(overlaps)*1.3
    fig, ax = plt.subplots(figsize=figsize2)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')
    putil.make_overlap_plot(overlaps, tt, ymax*.9, ymax*.95, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_5D')

    # Fig 5E
    coeffs = {0: -0.4, 1: 0.6, 2: 0.4}
    overlaps = util.get_overlaps(coeffs, params)[:tk, 1:]
    ymax = np.max(overlaps)*1.3
    fig, ax = plt.subplots(figsize=figsize2)
    ax.set_ylabel(r'$q_{\mu}(t)$')
    ax.set_xlabel(r'$t$')
    putil.make_overlap_plot(overlaps, tt, ymax*.9, ymax*.95, ax)
    ax.set_ylim([None, ymax])
    putil.savefig(ax, 'fig_5E')

    plt.close("all")

# Fig 2
def verify_mf_plots(params, run_num, cmap=putil.overlap_cmap, fname_pre='fig_2'):
    if run_num is not None and run_num != 0:
        return -1
    params = copy.deepcopy(params)
    inp_params = params['inp_params']
    sim_params = params['sim_params']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    coeffs = {0: 0, 1: 1.5}
    P = inp_params['P']

    def make_plot(N, coeffs, fname):
        inp_params['N'] = N
        mf_soln = util.get_meanfield_from_coeffs(coeffs, params)[:, 1::5]
        overlaps = util.get_overlaps(coeffs, params)[:, 1::5]

        K = overlaps.shape[1]
        cycle_len = 6
        colors = cmap[:K]

        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylabel(r'$q_{\mu}(t)$')
        ax.set_xlabel(r'$t$')
        for k in range(K-1,-1,-1):
            ck = k % cycle_len
            lines1 = ax.plot(tt, overlaps[:, k], color=colors[ck])
            lines2 = ax.plot(tt, mf_soln[:, k], color=colors[ck],
                    linestyle='--')
        putil.savefig(ax, fname)

        figlegend = plt.figure(figsize=figsize1)
        figlegend.legend([lines1[0], lines2[0]], ["Network", "Mean Field"],
                         loc='center')
        fnameleg = (plotdir/'legend'/f'{fname}_legend').with_suffix(ext)
        figlegend.savefig(fnameleg, bbox_inches='tight', pad_inches=fig_pad)

    make_plot(5000, coeffs, fname_pre + 'A')
    make_plot(20000, coeffs, fname_pre + 'B')

    A = np.zeros((P,P))
    for k in range(10):
        A[k, k] = 0.1
        A[k+1, k] = 1.3
    for k in range(10, P-1):
        A[k, k] = -0.1
        A[k+1, k] = 1.5

    q0 = np.zeros(P)
    q0[0] = 1
    def make_plot(N, coeffs, fname):
        inp_params['N'] = N
        Gf = lambda x: util.G(x, sim_params['theta'], sim_params['sig'],
                         sim_params['rspan'])
        mf_soln = util.simulate_meanfield(q0, Gf, tt, A,
                                          sim_params['h_sig'])[:, 1::5]
        overlaps = util.get_overlaps(A, params)[:,1::5]

        K = overlaps.shape[1]
        cycle_len = 6
        colors = cmap[:K]

        fig, ax = plt.subplots(figsize=figsize2)
        ax.set_ylabel(r'$q_{\mu}(t)$')
        ax.set_xlabel(r'$t$')
        for k in range(K-1,-1,-1):
            ck = k % cycle_len
            lines1 = ax.plot(tt, overlaps[:, k], color=colors[ck])
            lines2 = ax.plot(tt, mf_soln[:, k], color=colors[ck],
                    linestyle='--')
        putil.savefig(ax, fname)

        figlegend = plt.figure(figsize=figsize1)
        figlegend.legend([lines1[0], lines2[0]], ["Network", "Mean Field"],
                         loc='center')
        fnameleg = (plotdir/'legend'/f'{fname}_2_legend').with_suffix(ext)
        figlegend.savefig(fnameleg, bbox_inches='tight', pad_inches=fig_pad)

    make_plot(5000, A, fname_pre + 'C')
    make_plot(20000, A, fname_pre + 'D')

# Fig S1
def G_plot(params):
    params = copy.deepcopy(params)
    sim_params = params['sim_params']
    xx = np.linspace(0, 1, 200)
    yy = util.G(xx, sim_params['theta'], sim_params['sig'], sim_params['rspan'])
    fig, ax = plt.subplots(figsize=(1.4, 1))
    ax.plot(xx, yy)
    ax.set_yticks(np.arange(0, 9, 2))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$G(x)$')
    putil.savefig(ax, 'fig_S1')

# Fig S2
def compare_tmu():
    def make_plot(coeffs, tmax, mu_min, mu_max, fname):
        P = int(mu_max*1.5)
        mus = np.arange(mu_min, mu_max+1)
        cbar = sum(list(coeffs.values()))
        coeffs = {key: val/cbar for key, val in coeffs.items()}
        coeffs = util.mp_round(coeffs)
        a=coeffs[1]+coeffs[-1]
        b=coeffs[1]-coeffs[-1]
        alpha = coeffs[1]-coeffs[-1] + 2*(coeffs[2]-coeffs[-2])
        beta = coeffs[1]+coeffs[-1] + 4*(coeffs[2]+coeffs[-2])
        a0 = coeffs[0]
        dt = .001
        tt = np.arange(0,tmax,dt)
        tl2_3 = int(2*len(tt)/3)
        tmu_true_full = util.periodic_tmus(tmax, dt, coeffs, P)

        tmu_sps = []
        tmu_trues = []
        for mu in mus:
            minmu = max(mu-40,0)
            maxmu = mu+15
            tmu_sp = util.saddlepnt_tmu(mu, tmax, dt, coeffs)
            tmu_sps.append(tmu_sp)
            tmu_trues.append(tmu_true_full[mu-1])
        tmu_theory2 = mus/alpha
        tmu_theory1 = mus/alpha - beta / (2*alpha**2)

        tmu_trues = np.array(tmu_trues)
        tmu_sps = np.array(tmu_sps)
        tmu_theory2 = np.array(tmu_theory2)
        tmu_theory1 = np.array(tmu_theory1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mus, tmu_trues-tmu_sps, label=r"$d/dt$ of saddle point")
        ax.plot(mus, tmu_trues-tmu_theory2, label=r"saddle point of $d/dt$")
        ax.plot(mus, tmu_trues-tmu_theory1, label="Taylor expansion")
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$t_{\mu}-\tilde{t}_{\mu}$')
        ax.legend()
        ax.set_xlim([0, mu_max+int(mu_max*.02)])
        putil.savefig(ax, fname + '_tmu')

        dmu_trues = np.diff(tmu_trues)
        dmu_sps = np.diff(tmu_sps)
        dmu_theory2 = np.diff(tmu_theory2)
        dmu_theory1 = np.diff(tmu_theory1)
        muss = mus[1:]
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(muss, dmu_trues-dmu_sps, label=r"$d/dt$ of saddle point")
        ax.plot(muss, dmu_trues-dmu_theory2, label=r"saddle point of $d/dt$")
        # ax.plot(muss, dmu_trues-dmu_theory1, label="Taylor expansion")
        ax.axhline(0, color='black', linestyle='--')
        ax.plot([0, mus[-1]], [dt, dt], color='black', linestyle='dotted',
                linewidth=.5)
        ax.plot([0, mus[-1]], [-dt, -dt], color='black', linestyle='dotted',
                linewidth=.5)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$d_{\mu}-\tilde{d}_{\mu}$')
        ax.legend()
        ax.set_xlim([0, mu_max+int(mu_max*.02)])
        putil.savefig(ax, fname + '_dmu')

    coeffs = {-2: 0, -1: -.3, 0: .1, 1: 1.8, 2: 0}
    make_plot(coeffs, 200, mu_min=5, mu_max=150, fname='fig_S2a_c')
    coeffs = {-2: 0, -1: .2, 0: .2, 1: .8, 2: 0}
    make_plot(coeffs, 200, mu_min=1, mu_max=50, fname='fig_S2b_d')

## Supplementary Figures S3 and S4
def limiting_cases(base_params, run_num=1, csum=1.5, fignames=None):
    if fignames is None:
        fignames = ['mu_plot', 'tmu_plot', 'pmu', 'inset', 'overlaps']
    assert len(fignames) == 5
    params = copy.deepcopy(base_params)
    coeff_list = make_coeffs_one_forward_const_g(csum)
    df = make_df(mu_data, coeff_list, params, 120, run_num)
    hue = make_coeff_combs(df)
    hue_order = putil.get_coeff_order(df, hue, 'linear')
    huevals = df[hue].unique()
    tmustr = r'$t_{\mu}$'
    dmustr = r'$d_{\mu}$'
    style = 'type'
    stylevals = df[style].unique()
    for hueval in huevals:
        dfb = df[hue]==hueval
        st_time = df[dfb&(df['type']=='network')]['peak time'].min()
        df[dfb&(df['type']=='approx')&(df['peak time']<=st_time)] = np.nan
    format_df(df, inplace=True)
    if len(df) == 0:
        return None
    l_integ = []
    baselinestr = r'$d_{\mu}$'
    for huev in huevals:
        for stylev in ['linear', 'network', 'mf']:
            baseline = df[(df[hue]==huev)&(df[style]=='approx')][dmustr].iloc[0]
            dfb = df[(df[hue]==huev)&(df[style]==stylev)]
            integral = ((dfb[dmustr]-baseline)*dfb[dmustr]).sum()
            l_integ.append({hue: huev, style: stylev,
                            'integral': np.abs(integral),
                            baselinestr: baseline})
            l_integ.append({hue: huev, style: 'unity',
                            'integral': baseline,
                            baselinestr: baseline})
    df_int = pd.DataFrame(l_integ)
    integ_norm_str = r'$D$'
    # integ_y = 'integral'
    integ_y = integ_norm_str
    df_int[integ_norm_str] = df_int['integral'] / df_int[baselinestr]

    # Fig S3a/S3c
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=df, ax=ax, x=r'$\mu$', y=r'$d_{\mu}$',
                     hue=hue, hue_order=hue_order, style='type',
                     style_order=['network', 'mf', 'linear', 'approx'], alpha=0.7)
    ax.set_ylim([-.04, None])
    putil.savefig(ax, fignames[0], ncols=1)

    # Fig S3b/S3d
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=df, ax=ax, x=r'$t_{\mu}$', y=r'$d_{\mu}$',
                     hue=hue, hue_order=hue_order, style='type',
                     style_order=['network', 'mf',  'linear', 'approx'], alpha=0.7)
    ax.set_ylim([-.04, None])
    putil.savefig(ax, fignames[1], ncols=1)

    # Fig S4a/S4b
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=df, ax=ax, x=r'$\mu$', y=r'$p_{\mu}$',
                     hue=hue, hue_order=hue_order, style='type',
                     style_order=['network', 'mf', 'linear'], alpha=0.7)
    ax.set_ylim([-.04, None])
    putil.savefig(ax, fignames[2], ncols=1)

    # Fig S3b/S3d inset
    fig, ax = plt.subplots(figsize=(1.7*.35, 1*.35))
    g = sns.lineplot(data=df_int, ax=ax, x=baselinestr, y=integ_y,
                     style='type',
                     style_order=['mf'],
                     alpha=0.7)
    ax.set_ylim([-.04, None])
    putil.savefig(ax, fignames[3], ncols=1)

            
