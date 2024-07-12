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
     # 'legend.fontsize': 'xx-small',
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

# Coefficient plots
dtype = ['network', 'mf_linear']
dtypestr = '_'.join(dtype)
base_params = {'inp_params': dict(S=1, P=100, seed=33,
                             N=35000,
                             input_type='gaussian',),
               'sim_params': dict(sig=0.1, theta=0,
                             rspan=2, rcenter=0,
                             T=60,
                             t_steps=800,
                             h_sig=0, seed=13)
         }


## Fig 1b and 1c
cp.model_overview_plots(base_params, run_num=run_num)
## Fig 2
cp.verify_mf_plots(base_params, run_num)
## Fig 3
cp.one_forward_plots(base_params, run_num)
## Fig 4
cp.ofob_plots(base_params, run_num)
## Fig 5
cp.two_forward_plots(base_params, run_num)
if run_num == 0:
    ## Fig S2
    cp.compare_tmu()
    ## Figure S1
    cp.G_plot(base_params)


# Exponential kernel plots

params_dsided = {
    'w_params': dict(type='double_exp', mag1=2, mag2=2, tau1=.25, tau2=1,
                     a=-20, b=20, offset=0),
    'inp_params': dict(S=1, P=100, seed=33,
                       N=40000,
                       T_xi=.5,
                       input_type='gaussian'),
    'sim_params': dict(
        vspan=1,
                       T=60, t_steps=800,
                       seed=13, h_sig=0, rspan=2,
                       rcenter=0, theta=0, sig=0.1)
      }

## Fig 6b, 6c
kp.Txi_plots(params_dsided, run_num)

# Parameter combination plots
figsize = (1.5, 1)
ylims = (0, 3)
params_combos = {'w_params': dict(type='double_exp',
                         mag1=1, mag2=3,
                         tau1=.25, tau2=1,
                         a=-20, b=20, offset=0),
       'inp_params': dict(S=1, P=60, seed=33,
                          N=40000,
                          T_xi=3,
                          input_type='gaussian'),
       'sim_params': dict(
           vspan=1,
           sim_type='network',
                          T=60,
                          t_steps=800,
                          seed=13, h_sig=0,
                          rspan=2, rcenter=0,
                          theta=0, sig=0.1)
      }
## Fig 6d
kp.params_combos_plots(params_combos, run_num)

ps0 = copy.deepcopy(params_dsided)
ps0['inp_params']['N'] = 40000
ps0['inp_params']['P'] = 100
ps0['w_params'] = dict(type='double_exp',
                       tau1=.5,
                       mag1=2,
                       tau2=1,
                       mag2=2,
                       a=-20, b=20,
                       offset=0)

# Fig S1
kp.fast_and_slow_plots(ps0, run_num)


print("Simulations complete.", flush=True)
print("\n\n\n\n")
