import copy
import numpy as np

T = 60
t_steps = 800
params_base = dict(w_params=dict(), inp_params=dict(P=40, seed=33, N=30000),
                   sim_params=dict(vspan=1, sig=.1, T=60, t_steps=800),
                  )
params_mf = copy.deepcopy(params_base)
params_mf['sim_params'] = dict(sim_type='mf_approx3', vspan=1, sig=.1)
params_full_net = copy.deepcopy(params_base)
params_full_net['sim_params'].update(sim_type='network')

params_constw1 = copy.deepcopy(params_base)
params_constw1['w_params'] = dict(type='linear', slope=0, mag1=.2, a=0, b=2)
params_dexpw1 = copy.deepcopy(params_base)
params_dexpw1['w_params'] = dict(
    type='double_exp', mag1=-5, mag2=5, tau1=.1, tau2=.5,
    a=-10, b=10, offset=0)
params_gauss = copy.deepcopy(params_base)

params_mf_constw1_gauss = {}
params_mf_dexpw1_gauss = {}
params_full_net_constw1_gauss = {}
params_full_net_dexpw1_gauss = {}
for key in params_base.keys():
    params_mf_constw1_gauss[key] = (params_mf[key] | params_constw1[key] |
                                    params_gauss[key])
    params_mf_dexpw1_gauss[key] = (params_mf[key] | params_dexpw1[key] |
                                   params_gauss[key])
    params_full_net_constw1_gauss[key] = (params_full_net[key] |
                                          params_constw1[key] |
                                          params_gauss[key])
    params_full_net_dexpw1_gauss[key] = (params_full_net[key] |
                                         params_dexpw1[key] |
                                         params_gauss[key])

