import argparse
import coeff_plots as cp
import kernel_plots as kp
from matplotlib import pyplot as plt
import seaborn as sns
import copy
import sys

sns.set_palette('colorblind')

plt.rcParams.update({
     'axes.titlesize': 'small',
     'axes.labelsize': 'x-small',
     'xtick.labelsize': 'xx-small',
     'ytick.labelsize': 'xx-small',
     'axes.labelweight': 'light',
     'axes.linewidth': .4,
     'xtick.major.width': .4,
     'ytick.major.width': .4,
     'legend.borderpad': .2,
     'legend.handleheight': .4,
     'lines.linewidth': .8,
     'text.usetex': True,
     'pdf.fonttype': 42,
     'ps.fonttype': 42,
})

parser_outer = argparse.ArgumentParser(
    description='outer', add_help=False)
parser_outer.add_argument('--run-num',
                          type=int, default=None, metavar='N',
                          help='Run number.')
args_outer, remaining_outer = parser_outer.parse_known_args()
run_num = args_outer.run_num

############################
### Coefficient plots

base_params = {'inp_params': dict(S=1, P=100, seed=33,
                                  N=35000,
                                  input_type='gaussian', sparsity=1),
               'sim_params': dict(sig=0.1, theta=0,
                                  periodic=False,
                                  rspan=2, rcenter=0,
                                  T=60,
                                  t_steps=800,
                                  h_sig=0,
                                  seed=13)
         }

## Fig 1B and 1B
cp.model_overview_plots(base_params, run_num=run_num)
## Fig 2
params = copy.deepcopy(base_params)
# params['inp_params']['sparsity'] = 0.5
# cp.verify_mf_plots(params, run_num, fname_pre='fig_2_sparsity_0p5')
params['inp_params']['sparsity'] = 1
cp.verify_mf_plots(params, run_num, fname_pre='fig_2_sparsity_1')
## Fig 3
cp.one_forward_plots(base_params, run_num)
## Fig 4
cp.ofob_plots(base_params, run_num)
## Fig 5
cp.two_forward_plots(base_params, run_num)
if run_num == 0:
    ## Figure S1
    cp.G_plot(base_params)
    ## Fig S2
    cp.compare_tmu()

## Supplementary Figures S3 and S4
params = copy.deepcopy(base_params)
params['inp_params']['N'] = 35000
params['sim_params']['t_steps'] = 2*4800
params['inp_params']['P'] = 140
fignames = ['fig_S3a', 'fig_S3b', 'fig_S4a', 'fig_S3b_inset',
            'fig_S3_overlaps_1']
cp.limiting_cases(params, run_num, 1.5, fignames)
fignames = ['fig_S3c', 'fig_S3d', 'fig_S4b', 'fig_S3d_inset',
            'fig_S3_overlaps_2']
cp.limiting_cases(params, run_num, .6, fignames)

############################
### Exponential kernel plots

params_dsided = {
    'w_params': dict(wtype='double_exp', mag1=2, mag2=2, tau1=.25, tau2=1,
                     a=-20, b=20, offset=0),
    'inp_params': dict(S=1, P=100, seed=33,
                       N=40000,
                       T_xi=.5,
                       input_type='gaussian',
                       sparsity=1.,),
    'sim_params': dict(vspan=1,
                       T=40, t_steps=1600,
                       seed=13, h_sig=0, rspan=2,
                       rcenter=0, theta=0, sig=0.1, periodic=False)
      }

# Fig 6B, 6C
kp.Txi_plots(params_dsided, run_num)

# Parameter combination plots
figsize = (1.5, 1)
ylims = (0, 3)
params_combos = {
    'w_params':   dict(wtype='double_exp', mag1=1, mag2=3, tau1=.25, tau2=1,
                       a=-20, b=20, offset=0),
    'inp_params': dict(S=1, P=60, seed=33, N=40000, T_xi=3,
                       input_type='gaussian', sparsity=1.),
    'sim_params': dict( vspan=1, sim_type='network', T=60, t_steps=800,
                       seed=13, h_sig=0, rspan=2, rcenter=0, theta=0, sig=0.1,
                       periodic=False)
}

# Fig 6D
kp.params_combos_plots(params_combos, run_num)

ps0 = copy.deepcopy(params_dsided)
ps0['inp_params']['N'] = 40000
ps0['inp_params']['P'] = 100
ps0['w_params'] = dict(wtype='double_exp',
                       tau1=.25,
                       mag1=2,
                       tau2=1,
                       mag2=2,
                       a=-20, b=20,
                       offset=0)

## Fig S1
kp.fast_and_slow_plots(ps0, run_num)

## Fig S3e/S3f/S4c
params = copy.deepcopy(params_dsided)
params['inp_params']['N'] = 100000
params['sim_params']['t_steps'] = 4800
params['inp_params']['P'] = 80
fignames = ['fig_S3e', 'fig_S3f', 'fig_S4c', 'fig_S3f_inset', 'kernel_overlaps',
            'kernel_overlaps_lin']
kp.limiting_cases(params, [1, .5, .1], [40,40,40], 50, None, run_num, fignames)

## Fig S4d
params['w_params']['mag1'] = 4
params['w_params']['mag2'] = 4
fignames = ['not_used', 'not_used', 'fig_S4d', 'not_used', 'not_used',
            'not_used']
kp.limiting_cases(params, [1, .5, .1], [40,40,40], 50, None, run_num, fignames)

## Fig S5a/S5b/S6a
params = copy.deepcopy(params_dsided)
params['inp_params']['P'] = 40
params['inp_params']['N'] = 100000
params['sim_params']['tau'] = .01
params['sim_params']['T'] = .5
params['sim_params']['t_steps'] = 4800

params['w_params']['mag1'] = .273*4000
params['w_params']['tau1'] = .0168
params['w_params']['mag2'] = .777*4000
params['w_params']['tau2'] = .0168

Ts = [2, 1, .5, .25]
Txis = [.1,.06,.03,.01]
fignames = ['fig_S5a', 'fig_S5b', 'fig_S6a', 'fig_S5b_inset', 'not_used',
            'not_used']
kp.limiting_cases(params, Txis, Ts, 30, None, run_num, fignames, units='ms')

# Fig S5e
params['w_params']['mag1'] = .273*4000
params['w_params']['tau1'] = .0337
params['w_params']['mag2'] = .777*4000
params['w_params']['tau2'] = .0168

Ts = [2, 1, .5, .25]
Txis = [.1,.06,.03,.01]
fignames = ['not_used', 'not_used', 'not_used', 'not_used', 'fig_S5e',
            'not_used']
kp.limiting_cases(params, Txis, Ts, 30, None, run_num, fignames, units='ms')

## Double exponential, realistic, faithful to tutor signal
## Fig S5c/S5d/S6b
params['w_params']['mag1'] = .273*6000
av =  0.285714285714285
params['w_params']['tau1'] = .0168 * av
params['w_params']['mag2'] = .777*6000
params['w_params']['tau2'] = .0168 * av
params['sim_params']['T'] = 2
Ts = [4, 2, 1, .5]
Ns = [100000, 100000, 100000, 100000]
Txis = [.1, .06, .03, .01]
fignames = ['fig_S5c', 'fig_S5d', 'fig_S6b', 'fig_S5d_inset', 'not_used',
            'not_used']
kp.limiting_cases(params, Txis, Ts, 20, Ns, run_num,
                  fignames, units='ms')

## Double exponential, realistic, faithful to tutor signal, increased magnitude
## Fig S6c
params['w_params']['mag1'] = .273*12000
av =  0.285714285714285
params['w_params']['tau1'] = .0168 * av
params['w_params']['mag2'] = .777*12000
params['w_params']['tau2'] = .0168 * av
params['sim_params']['T'] = 2
Ts = [4, 2, 1, .5]
Ns = [100000, 100000, 100000, 100000]
Txis = [.1, .06, .03, .01]
fignames = ['not_used', 'not_used', 'fig_S6c', 'not_used', 'not_used',
            'not_used']
kp.limiting_cases(params, Txis, Ts, 20, Ns, run_num,
                  fignames, units='ms')

print("Simulations complete.", flush=True)
print("\n\n\n\n")

