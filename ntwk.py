from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix
import os
import time

from aux import save


# INITIALIZATION HELPERS

def spks_forced_rand(ntwk, mask, itvl, freq, dt):
    """
    Sample a forced spike matrix from a Poisson distribution.
    
    :param ntwk: LIFNtwk instance
    :param mask: mask over which cells to potentially make spike
    :param itvl: itvl to force spikes over
    :param freq: freq of forced spikes
    :param dt: simulation timestep
    """
    # generate spks inside itvl
    dur = itvl[1] - itvl[0]
    
    if mask.dtype == 'bool':
        mask = mask.nonzero()[0]
    
    spks_forced = np.zeros((int(dur/dt), ntwk.n), dtype=bool)
    spks_forced[:, mask] = np.random.binomial(
        1, freq*dt, (len(spks_forced), len(mask)))
    
    # buffer spks_forced to start at itvl[0]
    buf = np.zeros((int(itvl[0]/dt), ntwk.n), dtype=bool)
    
    return np.concatenate([buf, spks_forced])


# CONNECTIVITY

def join_w(targs, srcs, ws):
    """
    Combine multiple weight matrices specific to pairs of populations
    into a single, full set of weight matrices (one per synapse type).
    
    :param targs: dict of boolean masks indicating targ cell classes
    :param srcs: dict of boolean masks indicating source cell classes
    :param ws: dict of inter-population weight matrices, e.g.:
        ws = {
            'AMPA': {
                ('EXC', 'EXC'): np.array([[...]]),
                ('INH', 'EXC'): np.array([[...]]),
            },
            'GABA': {
                ('EXC', 'INH'): np.array([[...]]),
                ('INH', 'INH'): np.array([[...]]),
            }
        }
        note: keys given as (targ, src)
    
    :return: ws_full, a dict of full ws, one per synapse
    """
    # convert targs/srcs to dicts if given as arrays
    if not isinstance(targs, dict):
        targs_ = deepcopy(targs)
        targs = {
            cell_type: targs_ == cell_type for cell_type in set(targs_)
        }
    if not isinstance(srcs, dict):
        srcs_ = deepcopy(srcs)
        srcs = {
            cell_type: srcs_ == cell_type for cell_type in set(srcs_)
        }
        
    # make sure all targ/src masks have same shape
    targ_shapes = [mask.shape for mask in targs.values()]
    src_shapes = [mask.shape for mask in srcs.values()]
    
    if len(set(targ_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    if len(set(src_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    n_targ = targ_shapes[0][0]
    n_src = src_shapes[0][0]
    
    # make sure weight matrix dimensions match sizes
    # of targ/src classes
    for syn, ws_ in ws.items():
        for (targ, src), w_ in ws_.items():
            if not w_.shape == (targs[targ].sum(), srcs[src].sum()):
                raise Exception(
                    'Weight matrix for {}: ({}, {}) does not match '
                    'dimensionality specified by targ/src masks.')
        
    # loop through synapse types
    dtype = list(list(ws.values())[0].values())[0].dtype
    ws_full = {}
    
    for syn, ws_ in ws.items():
        
        w = np.zeros((n_targ, n_src), dtype=dtype)
        
        # loop through population pairs
        for (targ, src), w_ in ws_.items():
            
            # get mask of all cxns from src to targ
            mask = np.outer(targs[targ], srcs[src])
            
            assert mask.sum() == w_.size
            
            w[mask] = w_.flatten()
            
        ws_full[syn] = w
        
    return ws_full


# NETWORK CLASS AND FUNCTIONS

class LIFNtwk(object):
    """
    Network of leaky integrate-and-fire (LIF) neurons.
    All parameters should be given in SI units
    (i.e., time constants in seconds, potentials in volts).
    This simulation uses exponential
    synapses for all synapse types.
    In all weight matrices, rows index target, cols index source.
    
    :param t_m: membrane time constant (or 1D array)
    :param e_l: leak reversal potential (or 1D array)
    :param v_th: firing threshold potential (or 1D array)
    :param v_reset: reset potential (or 1D array)
    :param e_ahp: afterhyperpolarization (potassium) reversal potential
    :param t_ahp: afterhyperpolarization time constant
    :param w_ahp: afterhyperpolarization magnitude 
    :param t_r: refractory time
    :param es_syn: synaptic reversal potentials (dict with keys naming
        synapse types, e.g., 'AMPA', 'NMDA', ...)
    :param ts_syn: synaptic time constants (dict)
    :param ws_rcr: recurrent synaptic weight matrices (dict with keys
        naming synapse types)
    :param ws_up: input synaptic weight matrices from upstream inputs (dict)
    :param plasticity: dict of plasticity params with the following keys:
        'masks': synaptic dict of boolean arrays indicating which synapses in 
            ws_up are plastic, i.e., which synapses correspond to ST->PC cxns
        'w_pc_st_maxs': synaptic dict of max values for plastic weights
        'T_W': timescale of activity-dependent plasticity
        'T_C': timescale of PC spike-counter auxiliary variable
        'C_S': threshold for spike-count-based plasticity activation
        'B_C': slope of spike-count nonlinearity
    :param sparse: whether to convert weight matrices to sparse matrices for
        more efficient processing
    """
    
    def __init__(self,
            t_m, e_l, v_th, v_reset, t_r,
            e_ahp=0, t_ahp=np.inf, w_ahp=0,
            es_syn=None, ts_syn=None, ws_up=None, ws_rcr=None, 
            plasticity=None, sparse=True):
        """Constructor."""

        # validate arguments
        if es_syn is None:
            es_syn = {}
        if ts_syn is None:
            ts_syn = {}
        if ws_up is None:
            ws_up = {}
        if ws_rcr is None:
            ws_rcr = {}

        self.syns = list(es_syn.keys())
        
        # check weight matrices have correct dims
        shape_rcr = list(ws_rcr.values())[0].shape
        shape_up = list(ws_up.values())[0].shape
        
        self.n = shape_rcr[1]

        # fill in unspecified weight matrices with zeros
        for syn in self.syns:
            if syn not in ws_rcr:
                ws_rcr[syn] = np.zeros(shape_rcr)
            if syn not in ws_up:
                ws_up[syn] = np.zeros(shape_up)
        
        # check syn. dicts have same keys
        if not set(es_syn) == set(ts_syn) == set(ws_rcr) == set(ws_up):
            raise ValueError(
                'All synaptic dicts ("es_syn", "ts_syn", '
                '"ws_rcr", "ws_inp") must have same keys.'
            )

        if not all([w.shape[0] == w.shape[1] == self.n for w in ws_rcr.values()]):
            raise ValueError('All recurrent weight matrices must be square.')

        # check input matrices' have correct dims
        self.n_up = list(ws_up.values())[0].shape[1]

        if not all([w.shape[0] == self.n for w in ws_up.values()]):
            raise ValueError(
                'Upstream weight matrices must have one row per neuron.')

        if not all([w.shape[1] == self.n_up for w in ws_up.values()]):
            raise ValueError(
                'All upstream weight matrices must have same number of columns.')

        # check plasticity parameters
        if plasticity is not None:
            # make sure all parameters are given
            if set(plasticity) != {
                    'masks', 'w_pc_st_maxs', 'T_W', 'T_C', 'C_S', 'B_C'}:
                raise KeyError(
                    'Argument "plasticity" must contain the correct keys '
                    '(see LIFNtwk documentation).')
            # make sure there is one plasticity matrix for each synapse type
            if set(plasticity['masks']) != set(ws_up):
                for syn in ws_up:
                    if syn not in plasticity['masks']:
                        plasticity['masks'][syn] = np.zeros(
                            (self.n, self.n_up), bool)
            # make sure plasticity matrices are boolean and same size as ws_up
            for w in plasticity['masks'].values():
                if w.shape != (self.n, self.n_up):
                    raise ValueError(
                        'All matrices in "plasticity[\'masks\']" must have same '
                        'shape as matrices in "ws_up".')
                if w.dtype != bool:
                    raise TypeError(
                        'All matrices in "plasticity[\'masks\']" must be '
                        'logical arrays.')
            # make sure max weight values dict has correct synaptic keys
            if set(plasticity['w_pc_st_maxs']) != set(ws_up):
                raise KeyError(
                    'Argument "plasticity[\'w_pc_st_maxs\']" must contain same '
                    'keys (synapse types) as argument "ws_up".')
        
        # make sure v_reset is actually an array
        if isinstance(v_reset, (int, float, complex)):
            v_reset = v_reset * np.ones(self.n)
            
        # store network params
        self.t_m = t_m
        self.e_l = e_l
        self.v_th = v_th
        self.v_reset = v_reset
        self.t_r = t_r
        self.e_ahp = e_ahp
        self.t_ahp = t_ahp
        self.w_ahp = w_ahp
        
        self.es_syn = es_syn
        self.ts_syn = ts_syn
        
        self.plasticity = plasticity
        if plasticity is not None:
            self.ns_plastic = {
                syn: w.sum() for syn, w in plasticity['masks'].items()}
         
        if sparse:
            ws_rcr = {syn: csc_matrix(w) for syn, w in ws_rcr.items()}
            ws_up = {syn: csc_matrix(w) for syn, w in ws_up.items()}
            
        self.ws_rcr = ws_rcr
        self.ws_up_init = ws_up

    def run(
            self, spks_up, dt, vs_0=None, gs_0=None, g_ahp_0=None, i_ext=None,
            vs_forced=None, spks_forced=None, store=None, report_every=None):
        """
        Run a simulation of the network.
        :param spks_up: upstream spiking inputs (rows are time points, 
            cols are neurons) (should be non-negative integers)
        :param dt: integration time step for dynamics simulation
        :param vs_0: initial vs
        :param gs_0: initial gs (dict of 1-D arrays)
        :param g_ahp_0: initial g_ahp (1-D array)
        :param vs_forced: voltages to force at given time points (rows 
            are time points, cols are neurons)
        :param spks_forced: bool array of spikes to force at given time 
            points (rows are time points, cols are neurons)
        :return: network response object
        """

        # validate arguments
        if vs_0 is None:
            vs_0 = self.e_l * np.ones(self.n)
        if gs_0 is None:
            gs_0 = {syn: np.zeros(self.n) for syn in self.syns}
        if g_ahp_0 is None:
            g_ahp_0 = np.zeros(self.n)
            
        if vs_forced is None:
            vs_forced = np.zeros((0, self.n))
        if spks_forced is None:
            spks_forced = np.zeros((0, self.n), dtype=bool)
        
        if type(spks_up) != np.ndarray or spks_up.ndim != 2:
            raise TypeError('"inps_upstream" must be a 2D array.')

        if not spks_up.shape[1] == self.n_up:
            raise ValueError(
                'Upstream input size does not match size of input weight matrix.')

        if not vs_0.shape == (self.n,):
            raise ValueError(
                '"vs_0" must be 1-D array with one element per neuron.')

        if not all([gs.shape == (self.n,) for gs in gs_0.values()]):
            raise ValueError(
                'All elements of "gs_0" should be 1-D array with '
                'one element per neuron.')
        if not g_ahp_0.shape == (self.n,):
            raise ValueError(
                '"g_ahp_0" should be 1-D array with one element per neuron.')

        if store is None:
            store = {}
        
        if 'vs' not in store:
            store['vs'] = np.float64
        if 'spks' not in store:
            store['spks'] = bool
        if 'gs' not in store:
            store['gs'] = np.float64
        if 'g_ahp' not in store:
            store['g_ahp'] = store['gs']
        if 'ws_plastic' not in store:
            store['ws_plastic'] = np.float64
        if 'cs' not in store:
            store['cs'] = store['ws_plastic']
           
        for key, val in store.items():
            
            if key == 'vs':
                assert val in (None, float, np.float16, np.float64)
            elif key == 'gs':
                assert val in (None, float, np.float16, np.float64)
            elif key == 'g_ahp':
                assert val in (None, float, np.float16, np.float64)
            elif key == 'ws_plastic':
                assert val in (None, float, np.float16, np.float64)
            elif key == 'spks':
                assert val in (None, bool)
            
        # prepare smln
        ts = np.arange(len(spks_up)) * dt
                  
        # initialize membrane potentials, conductances, and refractory counters
        vs_prev = vs_0.copy()
        spks_prev = np.zeros(vs_0.shape, dtype=bool)
        gs_prev = {syn: gs_0[syn].copy() for syn in self.syns}
        g_ahp_prev = g_ahp_0.copy()
        rp_ctrs = np.zeros(self.n)
                  
        # allocate space for slmn results and store initial values
        sim_shape = (len(ts), self.n)
        
        vs = None
        spks = None
        gs = None
        g_ahp = None
        
        if (i_ext is None):
            i_ext = np.zeros(len(ts))
        
        if store['vs'] is not None:
            vs = np.nan * np.zeros(sim_shape, dtype=store['vs'])
            vs[0, :] = vs_prev.copy()
                  
        if store['spks'] is not None:
            spks = np.zeros(sim_shape, dtype=bool)
            spks[0, :] = spks_prev.copy()
                  
        if store['gs'] is not None:
            gs = {
                syn: np.nan * np.zeros(sim_shape, dtype=store['gs'])
                for syn in self.syns
            }
                  
            for syn in self.syns:
                gs[syn][0, :] = gs_0[syn].copy()
                  
        if store['g_ahp'] is not None:
            g_ahp = np.nan * np.zeros(sim_shape, dtype=store['g_ahp'])
            g_ahp[0, :] = g_ahp_0.copy()
        
        # initialize plasticity variables
        if self.plasticity is not None:
            
            # rename variables to make them more accessible
            masks_plastic = self.plasticity['masks']
            w_pc_st_maxs = self.plasticity['w_pc_st_maxs']
            t_w = self.plasticity['T_W']
            t_c = self.plasticity['T_C']
            c_s = self.plasticity['C_S']
            b_c = self.plasticity['B_C']
            
            # set initial values for plasticity and spk ctr
            ws_plastic_prev = {
                syn: self.ws_up_init[syn][mask]
                for syn, mask in masks_plastic.items()
            }
            # correct for shitty sparse matrix handling...
            for syn in self.syns:
                temp = np.nan * np.zeros(ws_plastic_prev[syn].shape)
                temp[:] = ws_plastic_prev[syn][:]
                ws_plastic_prev[syn] = temp.flatten()
                
            cs_prev = np.zeros(self.n)
            
            # allocate space for plasticity variables
            # NOTE: ws_plastic values are time-series of just the plastic weights
            # in a 2D array where rows are time points and cols are weights
            ws_plastic = None
            cs = None
            
            if store['ws_plastic'] is not None:
                dtype = store['ws_plastic']
                ws_plastic = {
                    syn: np.nan * np.zeros((len(ts), n_plastic), dtype=dtype)
                    for syn, n_plastic in self.ns_plastic.items()
                }
                  
                for syn, mask in masks_plastic.items():
                    ws_plastic[syn][0] = self.ws_up_init[syn][mask].copy()
                  
            if store['cs'] is not None:
                cs = np.zeros(sim_shape, dtype=store['cs'])
                cs[0] = 0
            
        else:
            masks_plastic = None
            ws_plastic = None
            cs = None
        
        # run simulation
        ws_up = deepcopy(self.ws_up_init)
        
        smln_start_time = time.time()
        last_update = time.time()
        
        for step in range(1, len(ts)):

            ## update dynamics
            for syn in self.syns:
                
                # calculate new conductances for all synapse types
                w_up = ws_up[syn]
                w_rcr = self.ws_rcr[syn]
                t_syn = self.ts_syn[syn]

                # calculate upstream and recurrent inputs to conductances
                inps_up = w_up.dot(spks_up[step])
                inps_rcr = w_rcr.dot(spks_prev)

                # decay conductances and add any positive inputs
                dg = -(dt/t_syn) * gs_prev[syn] + inps_up + inps_rcr
                gs_prev[syn] = gs_prev[syn] + dg
             
            # calculate new AHP inputs
            inps_ahp = self.w_ahp * spks_prev
            
            # decay ahp conductance and add new inputs
            dg_ahp = (-dt/self.t_ahp) * g_ahp_prev + inps_ahp
            g_ahp_prev = g_ahp_prev + dg_ahp
                  
            # calculate current input resulting from synaptic conductances
            ## note: conductances are relative, so is_g are in volts
            is_g = [
                gs_prev[syn] * (self.es_syn[syn] - vs_prev)
                for syn in self.syns
            ]
            
            # add in AHP current
            is_g.append(g_ahp_prev * (self.e_ahp - vs_prev))
            
            # get total input current
            is_all = np.sum(is_g, axis=0) + i_ext[step]
            
            # update membrane potential
            dvs = -(dt/self.t_m) * (vs_prev - self.e_l) + is_all
            vs_prev = vs_prev + dvs
            
            # force refractory neurons to reset potential
            vs_prev[rp_ctrs > 0] = self.v_reset[rp_ctrs > 0]
            
            # force vs if desired
            if step < len(vs_forced):
                mask = ~np.isnan(vs_forced[step])
                vs_prev[mask] = vs_forced[step][mask]
            
            # identify spks
            spks_prev = vs_prev >= self.v_th
                  
            # force extra spks if desired
            if step < len(spks_forced):
                spks_prev[spks_forced[step] == 1] = 1
                   
            # reset membrane potentials of spiking neurons
            vs_prev[spks_prev] = self.v_reset[spks_prev]
            
            # set refractory counters for spiking neurons
            rp_ctrs[spks_prev] = self.t_r[spks_prev]
            # decrement refractory counters for all neurons
            rp_ctrs -= dt
            # adjust negative refractory counters up to zero
            rp_ctrs[rp_ctrs < 0] = 0
            
            ## update plastic weights
            if self.plasticity is not None:
                
                # calculate and store updated spk-ctr
                cs_prev = update_spk_ctr(
                    spks=spks_prev, cs_prev=cs_prev, t_c=t_c, dt=dt)
                
                # calculate new weight values for each syn type
                for syn in self.syns:
                
                    # reshape spk-ctr variable to align with updated weights
                    cs_prev_syn = cs_prev[masks_plastic[syn].nonzero()[0]]
                    
                    # update weight values
                    ws_plastic_prev[syn] = update_plastic_weights(
                        cs=cs_prev_syn, ws_prev=ws_plastic_prev[syn],
                        c_s=c_s, b_c=b_c, t_w=t_w,
                        w_pc_st_max=w_pc_st_maxs[syn], dt=dt)

                # insert updated weights into ws_up
                for syn, mask in masks_plastic.items():
                    ws_up[syn][mask] = ws_plastic_prev[syn]
                  
            # store vs
            if store['vs'] is not None:
                vs[step] = vs_prev.copy()
            
            # store spks
            if store['spks'] is not None:
                spks[step] = spks_prev.copy()

            # store conductances
            if store['gs'] is not None:
                for syn in self.syns:
                    gs[syn][step] = gs_prev[syn].copy()
                  
            # store ahp conductance
            if store['g_ahp'] is not None:
                g_ahp[step] = g_ahp_prev.copy()

            if self.plasticity is not None:
                if store['ws_plastic'] is not None:
                    for syn in self.syns:
                        ws_plastic[syn][step] = ws_plastic_prev[syn].copy()
                  
                if store['cs'] is not None:
                    cs[step] = cs_prev.copy()
            
            if report_every is not None:
                if time.time() > last_update + report_every:
                    
                    print('{0}/{1} steps completed after {2:.3f} s...'.format(
                        step + 1, len(ts), time.time() - smln_start_time))
                    
                    last_update = time.time()
        
        if self.plasticity is not None:
            if store['ws_plastic'] is None:
                ws_plastic = {
                    syn: ws_plastic_prev[syn].flatten()[None, :]
                    for syn in self.syns
                }
                
        # return NtwkResponse object
        return NtwkResponse(
            ts=ts, vs=vs, spks=spks, v_rest=self.e_l, v_th=self.v_th,
            gs=gs, g_ahp=g_ahp, ws_rcr=self.ws_rcr, ws_up=self.ws_up_init,
            cs=cs, ws_plastic=ws_plastic, masks_plastic=masks_plastic)


def z(c, c_s, b_c):
    return 1 / (1 + np.exp(-(c - c_s)/b_c))


def update_spk_ctr(spks, cs_prev, t_c, dt):
    """
    Update the spk-ctr auxiliary variable.
    :param spks: multi-unit spk vector from current time step
    :param cs_prev: spk-ctrs for all cells at previous time step
    :param t_c: spk-ctr time constant (see parameters.ipynb)
    :param dt: numerical integration time step
    """
    dc = -cs_prev * dt / t_c + spks.astype(float)

    return cs_prev + dc


def update_plastic_weights(cs, ws_prev, c_s, b_c, t_w, w_pc_st_max, dt):
    """
    Update the plastic cxns from ST to PC.
    
    :param cs: spk-ctrs for all cells at current time step
    :param ws_prev: 1-D array of plastic weight values at previous timestep
    :param c_s: spk-ctr threshold (see parameters.ipynb)
    :param b_c: spk-ctr nonlinearity slope (see dynamics.ipynb)
    :param t_w: weight change timescale (see dynamics.ipynb)
    :param w_pc_st_max: syn-dict of maximum ST->PC weight values
    :param dt: numerical integration time step
    """
    if cs.shape != ws_prev.shape:
        raise ValueError(
            'Spk-ctr "cs" and plastic weights "ws_prev" must have same shape.')
        
    dw = z(cs, c_s, b_c) * (w_pc_st_max - ws_prev) * dt / t_w 
    return ws_prev + dw


class NtwkResponse(object):
    """
    Class for storing network response parameters.
    :param ts: timestamp vector
    :param vs: membrane potentials
    :param spks: spk times
    :param gs: syn-dict of conductances
    :param ws_rcr: syn-dict of recurrent weight matrices
    :param ws_up: syn-dict upstream weight matrices
    :param cell_types: array-like of cell-types
    :param cs: spk ctr variables for each cell
    :param ws_plastic: syn-dict of time-courses of plastic weights
    :param masks_plastic: syn-dict of masks specifying which weights the plastic
        ones correspond to
    :param pfcs: array of cell place field centers
    """

    def __init__(
            self, ts, vs, spks, v_rest, v_th, gs, g_ahp, ws_rcr, ws_up, 
            cell_types=None, cs=None, ws_plastic=None, masks_plastic=None,
            pfcs=None):
        """Constructor."""
        # check args
        if (cell_types is not None) and (len(cell_types) != vs.shape[1]):
            raise ValueError(
                'If "cell_types" is provided, all cells must have a type.')
            
        self.ts = ts
        self.vs = vs
        self.spks = spks
        self.v_rest = v_rest
        self.v_th = v_th
        self.gs = gs
        self.g_ahp = g_ahp
        self.ws_rcr = ws_rcr
        self.ws_up = ws_up
        self.cell_types = cell_types
        self.cs = cs
        self.ws_plastic = ws_plastic
        self.masks_plastic = masks_plastic
        self.pfcs = pfcs
        
        self.dt = np.mean(np.diff(ts))
        self.fs = 1 / self.dt

    def save(self, save_file, save_gs=False, save_ws=True, save_place_fields=True):
        """
        Save network response to file.
        :param save_file: path of file to save it to (do not include .db extension)
        :param save_gs: whether to save conductances
        :param save_ws: whether to save connectivity matrices
        :param save_positions: whether to save positions
        """
        data = {
            'ts': self.ts,
            'vs': self.vs,
            'spks': self.spks,
            'v_rest': self.v_rest,
            'v_th': self.v_th,
            'cell_types': self.cell_types,
        }

        if save_gs:
            data['gs'] = self.gs
            data['g_ahp'] = self.g_ahp

        if save_ws:
            data['ws_rcr'] = self.ws_rcr
            data['ws_up'] = self.ws_up
            data['ws_plastic'] = self.ws_plastic
            data['masks_plastic'] = self.masks_plastic

        if save_place_fields:
            data['pfcs'] = self.pfcs
        
        data['dt'] = self.dt
        data['fs'] = self.fs

        return save(save_file, data)
    
    @property
    def n(self):
        """Number of neurons."""
        return self.vs.shape[1]