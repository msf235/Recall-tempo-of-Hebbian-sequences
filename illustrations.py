
from matplotlib import pyplot as plt
import pickle as pkl
import matplotlib.ticker as ticker
import numpy as np
import scipy
from scipy import special
from pathlib import Path
from joblib import Memory
import time
import plots_utils as util
import itertools
import pandas as pd
from scipy.signal import ricker
import seaborn as sns
import model_output_manager_hash as mom

plt.rcParams.update({
     'axes.titlesize': 'small',
     'axes.labelsize': 'small',
     'xtick.labelsize': 'x-small',
     'ytick.labelsize': 'x-small',
     'axes.labelweight': 'light',
     'axes.linewidth': .4,
     'xtick.major.width': .4,
     'ytick.major.width': .4,
     # 'legend.fontsize': 'xx-small',
     'legend.borderpad': .2,
     'legend.handleheight': .4,
     'lines.linewidth': .8,
     'text.usetex': True,
})

n1 = 3
np.random.seed(2)

def rand_patts():
    V = np.random.randn(3, n1)
    ps = dict(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,      
        top=False,         # ticks along the top edge are off
        right=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    # plt.tick_params(**dict)
    # fig, ax = plt.subplots(figsize=(.25,1))
    fig, ax = plt.subplots(figsize=(.125,.5))
    ax.imshow(V[0].reshape(n1, 1), aspect='equal')
    plt.tick_params(**ps)
    fig.savefig('plots/v1.pdf', bbox_inches='tight', pad_inches=0.0)

    fig, ax = plt.subplots(figsize=(.125,.5))
    ax.imshow(V[1].reshape(n1, 1), aspect='equal')
    plt.tick_params(**ps)
    fig.savefig('plots/v2.pdf', bbox_inches='tight', pad_inches=0.0)

    fig, ax = plt.subplots(figsize=(.125,.5))
    ax.imshow(V[2].reshape(n1, 1), aspect='equal')
    plt.tick_params(**ps)
    fig.savefig('plots/v3.pdf', bbox_inches='tight', pad_inches=0.0)

def plot_gfun(rmax, theta, sigma):
    def gfun(x):
        return np.exp(-theta**2/(2*(sigma**2+x))) \
                / np.sqrt(2*np.pi*(sigma**2+x))
    xx = np.linspace(0, 1, 300)
    gg = gfun(xx)
    fig, ax = plt.subplots(figsize=(1.5, 1))
    ax.plot(xx, gg)
    fig.savefig('plots/gfun.pdf', bbox_inches='tight', pad_inches=0.0)

def ricker(x, a):
    A = 2/(np.sqrt(3*a)*(np.pi**0.25))
    return A * (1 - (x/a)**2) * np.exp(-0.5*(x/a)**2)
    

def plot_w():
    
    x = np.linspace(-.25, .5, 500) 
    # Generate the Ricker wavelet
    # w = ricker(x, .1)
    def wfun(x):
        return -5*ricker(x, .1)*(x>0)
    coeffs = util.compute_coeffs(wfun, 0.5, 10)
    coeffs = {k: round(v,3) for k, v in coeffs.items() if round(v,3)!=0}
    print(coeffs)

    # Plot the Ricker wavelet
    fig, ax = plt.subplots(figsize=(1.5, 1))
    ax.plot(x, wfun(x), color='red', linewidth=3)
    fig.savefig('plots/wfun1.pdf', bbox_inches='tight', pad_inches=0.0)
    # plt.show()



if __name__ == '__main__':
    # rand_patts()
    # plot_gfun(1, 0, .1)
    plot_w()
