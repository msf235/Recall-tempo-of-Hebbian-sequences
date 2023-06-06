import pickle as pkl
import itertools
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
from scipy import special, integrate
from scipy.signal import find_peaks
from pathlib import Path
import time
import model_output_manager_hash as mom

memory = mom.Memory('make_plots_cache')
# memory.clear()
memory_overlaps = mom.Memory('overlaps_cache')
# memory_overlaps.clear()

def mp_round(coeff_dict):
    """Round all values in coeff_dict to less than machine precision, so that
    memoization will transfer across machines."""
    ret_dict = {key: round(val, 10) for key, val in coeff_dict.items()}
    return ret_dict

# Xil = []
# def process_patterns(S, P, N, input_type, seed, reset=False):
    # if len(Xil) == 0 or reset:
        # Xil.clear()
        # rng = np.random.default_rng(seed*7)
        # if input_type == 'binary_01':
            # mean = .5
            # Xi = (rng.random(size=(S, P, N)) > .5).astype(float)
        # elif input_type == 'gaussian':
            # mean = 0
            # Xi = rng.normal(size=(S, P, N))
        # else:
            # raise AttributeError("input_type not recognized.")
        # Xil.append(Xi)

def make_patterns(S, P, N, input_type, seed):
    rng = np.random.default_rng(seed*7)
    if input_type == 'binary_01':
        mean = .5
        Xi = (rng.random(size=(S, P, N)) > .5).astype(float)
        return Xi
    elif input_type == 'gaussian':
        mean = 0
        Xi = rng.normal(size=(S, P, N))
        return Xi
    else:
        raise AttributeError("input_type not recognized.")

def phi(x, rspan=1, theta=0, sig=.1, rcenter=1):
    return rspan * .5 * (rcenter + special.erf((x - theta)/(sig * 2**.5)))

def G(x, theta, sig, rspan):
    c1 = 2*(sig**2 + x)
    c2 = 1/np.sqrt(np.pi*c1)
    return rspan*c2*np.exp(-theta**2 / c1)

def get_peaks(x, tt):
    max_time_idx = np.argmax(x, axis=0).tolist()
    maxs = np.max(x, axis=0).tolist()
    peaks = (np.asarray(tt)[max_time_idx]).tolist()
    return peaks, maxs, max_time_idx

def make_weights_old(offset, cs, Xic):
    """Make network weights from coefficient vector cs.
    The full coefficient matrix is toeplitz so that a coefficient vector
    is sufficient to specify it."""
    S, P, N = Xic.shape
    if not hasattr(cs, '__len__'):
        cs = cs*np.ones(P)
    Xi_shifted = np.roll(Xic, -offset, axis=1)
    Xic = cs.reshape(1, P, 1) * Xic
    if offset > 0:
        Xi_trunc = Xic[:, :P-offset]
        Xi_shifted_trunc = Xi_shifted[:, :P-offset]
    elif offset < 0:
        Xi_trunc = Xic[:, -offset:]
        Xi_shifted_trunc = Xi_shifted[:, -offset:]
    else:
        Xi_trunc = Xic
        Xi_shifted_trunc = Xi_shifted
    cov = Xi_shifted_trunc.transpose(0, 2, 1) @ Xi_trunc
    W = cov.sum(0) / N
    return W

def make_weight_term(offset, Xic):
    """Make term in weight matrix corresponding to offset.
    The full coefficient matrix is toeplitz so that a coefficient vector
    is sufficient to specify it."""
    S, P, N = Xic.shape
    Xi_shifted = np.roll(Xic, -offset, axis=1)
    if offset > 0:
        Xi_trunc = Xic[:, :P-offset]
        Xi_shifted_trunc = Xi_shifted[:, :P-offset]
    elif offset < 0:
        Xi_trunc = Xic[:, -offset:]
        Xi_shifted_trunc = Xi_shifted[:, -offset:]
    else:
        Xi_trunc = Xic
        Xi_shifted_trunc = Xi_shifted
    cov = Xi_shifted_trunc.transpose(0, 2, 1) @ Xi_trunc
    Woffset = cov.sum(0) / N
    return Woffset

def make_weights(Xi, cvd):
    """Make weights."""
    tic = time.time()
    print("Making weights.", flush=True)
    cmax = max(cvd.keys())
    W = 0
    for k, c in cvd.items():
        W += c*make_weight_term(k, Xi)
        print(f"Finished weight {k} of {cmax}", flush=True)
    toc = time.time()
    elaps = toc-tic
    print(f"Made weights in {elaps} s", flush=True)
    return W

def make_weights_full(Xi, A, mean=0):
    """Make network weights from coefficient matrix A."""
    S, P, N = Xi.shape
    W = np.zeros((N, N))
    for k1 in range(P):
        for k2 in range(P):
            if A[k1, k2] != 0:
                print(k1, k2, flush=True)
                W += A[k1, k2] * np.outer(Xi[0,k1]-mean, Xi[0,k2]-mean)
    W = W / N
    return W

def get_weights(Xi, coeffs):
    if isinstance(coeffs, dict):
        W = make_weights(Xi, coeffs)
    else:
        W = make_weights_full(Xi, coeffs)
    return W

def make_weights_mf(coeffs, P, periodic=False):
    """Make Toeplitz coefficient matrix from coefficient dictionary."""
    coeffs_full = np.zeros((P,))
    for k1, val in coeffs.items():
        coeffs_full[k1] = val
    min_bndy = min(coeffs.keys())
    max_bndy = max(coeffs.keys())
    W = np.zeros((P,P))
    for k in range(P):
        csv_rolled = np.roll(coeffs_full, k)
        if not periodic:
            m1 = P+min_bndy+k
            csv_rolled[m1:] = 0
            m2 = max(0, -P+max_bndy+k+1)
            csv_rolled[:m2] = 0
        W[k] = csv_rolled
    return W.T.copy()

def quad_one_way(f, a, b):
    if a >= b:
        return 0
    return integrate.quad(f, a, b, full_output=True)[0]

def compute_coeffs(w, T_xi, P):
    """Make coefficient dictionary by convolving Hebbian kernel w
    over patterns with duration T_xi."""
    csvd = {}
    for k in range(-P+1, P, 1):
        a = k*T_xi
        b = (k+1)*T_xi
        if k == 0:
            def inner(t):
                return quad_one_way(w, a-t, 0) + quad_one_way(w, 0, b-t)
        else:
            def inner(t):
                return quad_one_way(w, a-t, b-t)
        temp = quad_one_way(inner, 0, T_xi)
        if temp != 0:
            csvd[k] = temp
    return csvd

def compute_A(w, T_xi_list, periodic=False):
    """Make coefficient matrix by convolving Hebbian kernel w
    over patterns with durations contained in T_xi_list."""
    P = len(T_xi_list)
    A = np.zeros((P, P))
    T_tt = np.zeros(P+1)
    T_tt[1:] = np.cumsum(T_xi_list)
    for mu in range(P):
        print(mu, flush=True)
        for k in range(P):
            a = T_tt[k]
            b = T_tt[k+1]
            def inner(t):
                return quad_one_way(w, a-t, b-t)
            temp = quad_one_way(inner, T_tt[mu], T_tt[mu+1])
            A[k, mu] = temp
    return A

def lin_ode_soln(q0, tt, A, t_factor=10):
    tt_int = np.linspace(tt[0], tt[-1], t_factor*len(tt))
    dt = tt_int[1]-tt_int[0]
    qs = np.zeros((len(tt), len(q0)))
    qs = np.zeros((len(tt), len(q0)))
    q = q0
    qs[0] = q
    k2 = 1
    for k in range(1, len(tt_int)):
        q = q + dt*A@q
        if k % t_factor == 0:
            qs[k2] = q
            k2 += 1
    return qs

def mf_soln(q0, tt, cvd, theta, sig, rspan):
    """Simulate (nonlinear) mean field equations and return solution."""
    print("simulating mean field equations.", flush=True)
    dt = tt[1]-tt[0]
    q_now = q0
    qs = np.zeros((len(tt), len(q0)))
    qs[0] = q_now
    A = make_weights_mf(cvd, P)
    for k in range(1, len(tt)):
        t1 = 0
        for key, c in cvd.items():
            qsc = q_now.copy()
            if key > 0:
                qsc[-key:] = 0
            else:
                qsc[:abs(key)] = 0
            t1 += c * qsc
        Gargs = (t1**2).sum()
        Gv = G(Gargs, theta, sig, rspan)
        dq_dt = -q_now + Gv * (A @ q_now)
        q_now = q_now + dt * dq_dt
        qs[k] = q_now
    return qs

def simulate_rnn(r0, phi, tt, W, seed, h_sig=0):
    """Simulate full network equations and return solution."""
    print("simulating.", flush=True)
    rng = np.random.default_rng(seed*13)
    dt = tt[1]-tt[0]
    r_soln = np.zeros((len(tt), len(r0)))
    r_now = r0
    r_soln[0] = r_now
    for k in range(1, len(tt)):
        hnoise = h_sig * rng.normal(size=len(r0))
        h = W@r_now + hnoise
        dr_dt = (-r_now + phi(h))
        r_now = r_now + dt * dr_dt
        r_soln[k] = r_now
    return r_soln


@memory_overlaps.cache
def simulate_rnn_subset(coeffs, params, k, save_weights=True):
    """Simulate full network equations and return a subset of the
    solutions. Subset size is set by k."""
    sim_params = params['sim_params']
    inp_params = params['inp_params']
    # process_patterns(inp_params['S'], inp_params['P'], inp_params['N'],
                   # inp_params['input_type'], inp_params['seed'])
    Xi = make_patterns(inp_params['S'], inp_params['P'], inp_params['N'],
                       inp_params['input_type'], inp_params['seed'])
    W = get_weights(Xi, coeffs)
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    dt = tt[1]-tt[0]
    r0 = Xi[0, 0]
    phil = lambda x: phi(x, sim_params['rspan'], sim_params['theta'],
                         sim_params['sig'], sim_params['rcenter'])
    r_soln = simulate_rnn(r0, phil, tt, W, sim_params['seed'])
    return r_soln[:, :k]


@memory_overlaps.cache 
def get_overlaps(coeffs, params):
    # Todo: add mean as an argument
    inp_params = params['inp_params']
    sim_params = params['sim_params']
    # process_patterns(inp_params['S'], inp_params['P'], inp_params['N'],
                   # inp_params['input_type'], inp_params['seed'])
    Xi = make_patterns(inp_params['S'], inp_params['P'], inp_params['N'],
                       inp_params['input_type'], inp_params['seed'])
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    dt = tt[1]-tt[0]
    r0 = Xi[0, 0]
    phil = lambda x: phi(x, sim_params['rspan'], sim_params['theta'],
                         sim_params['sig'], sim_params['rcenter'])
    W = get_weights(Xi, coeffs)
    rs = simulate_rnn(r0, phil, tt, W, sim_params['seed'], sim_params['h_sig'])
    # qs = (rs@Xil[0][0].T)/inp_params['N']
    qs = (rs@Xi[0].T)/inp_params['N']
    return qs

def get_overlaps2_periodic(tt, coeffs, P):
    cbar = sum(list(coeffs.values()))
    coeffs = {key: val/cbar for key, val in coeffs.items()}
    A = -np.eye(P,P) + make_weights_mf(coeffs, P, periodic=True)
    q0 = np.zeros(P)
    q0[0] = 1
    qs = lin_ode_soln(q0, tt, A)
    return qs

@memory_overlaps.cache
def periodic_tmus(tmax, dt, coeffs, P, prominence=.001):
    tt = np.arange(0, tmax+dt, dt)
    qsfull = get_overlaps2_periodic(tt, coeffs, P)
    tmus = np.zeros(P)
    for mu in range(1, P):
        peaks = find_peaks(qsfull[:,mu], prominence=prominence)[0]
        if len(peaks) > 0:
            tmu = tt[peaks[0]]
        else:
            tmu = np.nan
        tmus[mu-1] = tmu
    return tmus

def overlaps_saddlepnt(tt, coeffs, mu):
    cbar = sum(list(coeffs.values()))
    coeffs = {key: val/cbar for key, val in coeffs.items()}
    a=coeffs[1]+coeffs[-1]
    b=coeffs[1]-coeffs[-1]
    a0 = coeffs[0]
    c = mu/tt
    d1 = (a**2 - b**2 + c**2).astype(complex)
    sd1 = np.sqrt(d1)
    phim = np.log((c-sd1)/(b-a))*c + sd1
    phiddp = -1j * sd1
    phiddm = 1j * sd1
    thetap = -np.pi/2
    thetam = 0
    sola = np.exp((-1+a0)*tt)/np.sqrt(2*np.pi*tt*np.abs(sd1))
    solb = np.exp(tt*phim + 1j*thetam)
    return np.real(sola*solb)

@memory_overlaps.cache
def saddlepnt_tmu(mu, tmax, dt, coeffs, prominence=.001):
    tt = np.arange(0, tmax+dt, dt)
    qs = overlaps_saddlepnt(tt, coeffs, mu)
    peaks = find_peaks(qs, prominence=prominence)[0]
    if len(peaks) == 0:
        return np.nan
    return tt[peaks[0]]

def get_data_network(coeffs, params, save_weights=True):
    inp_params = params['inp_params']
    sim_params = params['sim_params']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    dt = tt[1]-tt[0]
    qs = get_overlaps(coeffs, params)[:, 1:]
    peaks, peakmags = np.array(get_peaks(qs, tt)[:2])
    cvds = {str(key): val for key, val in coeffs.items()}
    temp = np.where(peaks>=tt[-1])[0]
    if len(temp) > 0:
        fin_peak = temp[0] - 1
    else:
        fin_peak = -1
    peaks = peaks[:fin_peak]
    diffs = np.nan * np.ones(peaks.shape)
    diffs[:-1] = np.diff(peaks)
    ds = []
    # breakpoint()
    cvds = {str(key): val for key, val in coeffs.items()}
    ds += [{'mu': k+1, 'peak time': peaks[k], 'peak diff': diffs[k],
            'mag': peakmags[k], 'type': 'network',
            'h_sig': sim_params['h_sig'], **cvds} for k in range(len(peaks))]
    df = pd.DataFrame(ds)
    return ds

def get_data_mf_approx(coeffs, params):
    inp_params = params['inp_params']
    P = inp_params['P']
    sim_params = params['sim_params']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    dt = tt[1]-tt[0]
    k = len(coeffs)
    s = list(coeffs.keys())[0]
    if s < 0:
        s = 'm' + str(abs(s))
    gapprox = 1/(sum(coeffs.values()))
    A = -np.eye(P,P) + gapprox * make_weights_mf(coeffs, P)
    q0 = np.zeros(P)
    q0[0] = 1
    qs = lin_ode_soln(q0, tt, A)[:, 1:]
    # qs = mf_soln1(q0, tt, cvd, theta, sig)[:, 1:]
    peaks, peakmags = np.array(get_peaks(qs, tt)[:2])
    temp = np.where(peaks>=tt[-1])[0]
    if len(temp) > 0:
        fin_peak = temp[0] - 1
    else:
        fin_peak = -1
    peaks = peaks[:fin_peak]
    # breakpoint()
    ds = []
    cvds = {str(key): val for key, val in coeffs.items()}
    diffs = np.nan * np.ones(peaks.shape)
    diffs[:-1] = np.diff(peaks)
    ds = []
    cvds = {str(key): val for key, val in coeffs.items()}
    ds += [{'mu': k+1, 'peak time': peaks[k], 'peak diff': diffs[k],
            'mag': peakmags[k], 'type': 'linear', **cvds}
           for k in range(len(peaks))]
    return ds

def simulate_meanfield(q0, G, tt, A, h_sig=0):
    q_now = q0
    qvs = np.zeros((len(tt), len(q0)))
    dt = tt[1] - tt[0]
    qvs[0] = q_now
    for k in range(1, len(tt)):
        qh = A@q_now
        Gv = G(qh@qh + h_sig**2)
        dq_dt = -q_now + Gv * qh
        q_now = q_now + dt * dq_dt
        qvs[k] = q_now
    return qvs

def get_meanfield_from_coeffs(coeffs, params):
    inp_params = params['inp_params']
    P = inp_params['P']
    sim_params = params['sim_params']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    dt = tt[1]-tt[0]
    A = make_weights_mf(coeffs, P)
    q0 = np.zeros(inp_params['P'])
    q0[0] = 1
    Gf = lambda x: G(x, sim_params['theta'], sim_params['sig'],
                     sim_params['rspan'])
    qs = simulate_meanfield(q0, Gf, tt, A,
                            sim_params['h_sig'])
    return qs

@memory.cache
def get_gfun_mf_data_core(coeffs, params):
    inp_params = params['inp_params']
    sim_params = params['sim_params']
    P = inp_params['P']
    print("simulating.", flush=True)
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    dt = tt[1]-tt[0]
    q0 = np.zeros(P)
    q0[0] = 1
    q_now = q0
    Gargs = np.zeros(len(tt))
    Gvs = np.zeros(len(tt))
    A = make_weights_mf(coeffs, P)
    for k in range(1, len(tt)+1):
        t1 = 0
        for key, c in coeffs.items():
            qsc = q_now.copy()
            if key > 0:
                qsc[-key:] = 0
            else:
                qsc[:abs(key)] = 0
            t1 += c * qsc
        Garg = (t1**2).sum()
        Gargs[k-1] = Garg
        Gv = G(Garg, sim_params['theta'], sim_params['sig'], sim_params['rspan'])
        Gvs[k-1] = Gv
        dq_dt = -q_now + Gv * (A @ q_now)
        q_now = q_now + dt * dq_dt
    return Gvs, Gargs

def get_gfun_mf_data(coeffs, params):
    Gvs, Gargs = get_gfun_mf_data_core(coeffs, params)
    sim_params = params['sim_params']
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    cvds = {str(key): val for key, val in coeffs.items()}
    ds = []
    for k in range(len(tt)):
        ds.append({'t': tt[k], 'g(t)': Gvs[k], 'Garg': Gargs[k],
                   'type': 'mf', **cvds})
    return ds

def get_gfun_network_data(coeffs, params):
    sim_params = params['sim_params']
    qs = get_overlaps(coeffs, params)
    tt = np.linspace(0, sim_params['T'], sim_params['t_steps']+1)
    t1 = 0
    for k, c in coeffs.items():
        qsc = qs.copy()
        t1 += c * qsc
    Gargs = (t1**2).sum(axis=1)
    Gvs = G(Gargs, sim_params['theta'], sim_params['sig'], sim_params['rspan'])
    coeffs = {str(key): val for key, val in coeffs.items()}
    ds = []
    for k in range(len(tt)):
        ds.append({'t': tt[k], 'g(t)': Gvs[k], 'Garg': Gargs[k],
                   'type': 'network', **coeffs})
    return ds, qs

def get_alphas(coeffs):
    coeffs = coeffs.copy()
    coeffs_keys = list(coeffs.keys())
    csum = sum(coeffs.values())
    csvdn = {k: v/csum for k, v in coeffs.items()}
    kmax = max(coeffs_keys)
    alpha1 = 0
    for k in range(1, kmax+1):
        if -k not in csvdn:
            csvdn[-k] = 0
        alpha1 += k*(csvdn[k]-csvdn[-k])
    alpha2 = 0
    for k in range(1, kmax+1):
        alpha2 += k**2*(csvdn[k]+csvdn[-k])
    return alpha1, alpha2

