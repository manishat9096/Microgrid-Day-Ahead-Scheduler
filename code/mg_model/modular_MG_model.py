"""
Modular multi-bus microgrid dispatch model (Gurobi).

The only input is data: efficiency tables (Stage) and system structure (SystemSpec). An architecture is a spec, not a code path — AC and DC are two dicts of stages fed to the same builder. Adding or dropping a converter is editing a list; adding or dropping a component is editing a list; adding a second bus is naming it.

Every power path is the same object: an ordered chain of Stages between a terminal (grid / PV / battery / load) and a bus. A chain exposes exactly two reference planes, terminal-side and bus-side. Components only ever talk about terminal-side power (battery MWh, PV availability, metered import); buses only ever see bus-side power. That is what removes the multiply-on-discharge / divide-on-charge bookkeeping.

An uncontrollable Sink never enters the LP: its power is known, so its chain is
inverted exactly. A controllable one (exact=False) is decided like any other component.

SOS2 is per stage, off by default (Stage(..., sos2=True)). Without it a stage's lambda weights may mix non-adjacent breakpoints and ride the lower convex envelope of its loss curve, understating loss -- none of the tabulated curves in this project are convex. Any sos2 stage turns the LP into a MIP, so set it where the curve warrants it.

Usage
-----
    from components import Stage, SystemSpec, Import, Source, Storage, Sink
    from modular_MG_model import Scenario, build, build_and_solve

    # 1. one Stage per converter. loading/eta are the raw datasheet rows -- the
    #    zero-power breakpoint is added for you. `rated` is INPUT-referred, so a
    #    converter serving P must be rated at least P / eta.
    l7    = [.05, .10, .20, .30, .50, .75, 1.00]
    boost = Stage('pv_dcdc', 1.83, l7, [.905, .938, .963, .973, .979, .978, .976])
    inv   = Stage('pv_dcac', 1.83, l7, [.890, .932, .962, .972, .978, .977, .974])
    ...

    # 2. the plant. Built once and reused for every run -- no prices, no profiles,
    #    no SOC state in here. All fields are keyword-only. Stage lists always run
    #    terminal -> bus, and any length works, including [] for a direct connection.
    spec = SystemSpec(buses = ['ac'], components = [
        Import (name = 'grid', bus = 'ac', power = 10.0, stages = [trafo]),
        Source (name = 'pv',   bus = 'ac', power = 1.83, stages = [boost, inv]),
        Storage(name = 'bess', bus = 'ac', power = 5.0, capacity = 9.0,
                minimum_soc = 0.1, maximum_soc = 0.9,
                stages = [bess_dcdc, bess_dcac]),
        Sink   (name = 'load', bus = 'ac', power = 3.53, stages = [load_dcdc, load_acdc])])

    # 3. one Scenario per run: horizon, timestep, and the series each component needs.
    #    Import -> 'price', Source -> 'production', Storage -> 'soc_init' as a fraction (+ optional
    #    'soc_final'), Sink -> 'demand', MW per step (a scalar is broadcast).
    sc = Scenario(n_t = 96, dt = 0.25, data = {'grid': {'price': price_day},
                                               'pv'  : {'production': pv_day},
                                               'bess': {'soc_init': 0.5},
                                               'load': {'demand': 3.53}})

    # 4. solve
    res = build_and_solve(spec, sc, name = 'ACmicrogrid_design1')

    res['cost_eur']                          # objective, EUR
    res['components']['grid']['P_terminal']  # metered import, MW per step
    res['components']['bess']['SOC']         # MWh, length n_t + 1
    res['losses_MWh']['pv.pv_dcac']          # energy lost in one stage
    res['efficiency']                        # served / (imported + harvested)

    # build(spec, sc) instead returns (model, extractors, ctx) if you want to set
    # Gurobi parameters, write an LP file, or add constraints before optimizing.

Loop a single spec over many Scenarios to sweep days. Changing architecture is
editing stage lists: the DC variant of the spec above drops the PV inverter, the
BESS AC/DC and the load rectifier, and swaps the transformer for a central AC/DC.

Cost, hourly dispatch, energies and per-stage losses are determined. The raw
15-minute dispatch is not, when prices are constant within the hour -- any
redistribution inside an hour is cost-neutral, so the LP has many optima.

Stage lists are always ordered terminal -> bus. The builder walks them forward for export flows (PV generating, battery discharging, grid importing) and backward for import flows (battery charging, load being served).
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from dataclasses import dataclass, field

from .components import (Stage, Component, Import, Source, Storage, Sink, Link, SystemSpec)


# ----------------------------------------------------------------- run inputs

@dataclass
class Scenario:
    """Everything that changes between runs; the plant itself is in SystemSpec.

    data maps a component name to the run inputs that component needs:

        Scenario(n_t = 96, dt = 0.25, data = {
            'grid': {'price': price_day},
            'pv'  : {'production': pv_day},
            'bess': {'soc_init': 0.5},           # 'soc_final' optional
            'load': {'demand': 3.53}})

    Import needs 'price', Source needs 'production', Storage needs 'soc_init' (fraction of capacity) and may
    take 'soc_final', Sink needs 'demand' (MW per step); its `power` is only the connection rating.
    """
    n_t: int = 96
    dt: float = 0.25
    # ----- new change here
    date: str = None                 # label only, carried through to the results
    data: dict = field(default_factory=dict)

    def need(self, name, key):
        try:
            return self.data[name][key]
        except KeyError:
            raise KeyError(f"scenario is missing data['{name}']['{key}']") from None

    def opt(self, name, key, default=None):
        return self.data.get(name, {}).get(key, default)


# ------------------------------------------------------------- chain of stages

def _stage_keys(stages):
    """Per-position keys; a stage object may legitimately appear twice in a chain."""
    seen, keys = {}, []
    for s in stages:
        seen[s.name] = seen.get(s.name, 0) + 1
        keys.append(s.name if seen[s.name] == 1 else f'{s.name}#{seen[s.name]}')
    return keys


class _Chain:
    """Lambda blocks for a list of stages given in FLOW order.

    For each stage k:  sum(lam) == 1,  P_in = sum(lam*P_in_k),  P_out = sum(lam*P_out_k).
    Consecutive stages are coupled by  out(i) == in(i+1).
    An empty stage list is a direct connection: one free variable, in == out.
    """

    def __init__(self, m, T, stages, tag):
        self.m, self.T, self.tag = m, T, tag
        self.stages = list(stages)
        self.lam = []
        self.keys = _stage_keys(self.stages)
        for i, s in enumerate(self.stages):
            lam = m.addVars(T, s.K, lb=0.0, name=f'lam_{tag}_{self.keys[i]}')
            m.addConstrs((gp.quicksum(lam[t, k] for k in s.K) == 1 for t in T), name=f'lamsum_{tag}_{self.keys[i]}')
            # ----- new change here
            if s.sos2:                       # weights must be two ADJACENT breakpoints
                w = [k + 1 for k in s.K]     # distinct, nonzero ordering weights
                for t in T:
                    m.addSOS(GRB.SOS_TYPE2, [lam[t, k] for k in s.K], w)
            self.lam.append(lam)
        for i in range(len(self.stages) - 1):
            m.addConstrs((self.stage_out(i, t) == self.stage_in(i + 1, t) for t in T), name=f'chain_{tag}_{i}')
        self.direct = None
        if not self.stages:
            self.direct = m.addVars(T, lb=0.0, name=f'P_{tag}')

    def stage_in(self, i, t):
        s = self.stages[i]
        return gp.quicksum(self.lam[i][t, k] * s.P_in_k[k] for k in s.K)

    def stage_out(self, i, t):
        s = self.stages[i]
        return gp.quicksum(self.lam[i][t, k] * s.P_out_k[k] for k in s.K)

    def stage_loss(self, i, t):
        return self.stage_in(i, t) - self.stage_out(i, t)

    def head(self, t):
        """Power entering the chain."""
        return self.direct[t] if self.direct is not None else self.stage_in(0, t)

    def tail(self, t):
        """Power leaving the chain."""
        return self.direct[t] if self.direct is not None else self.stage_out(len(self.stages) - 1, t)

    def max_deliverable(self):
        """Largest power the chain can put out, given each stage's rated input."""
        P = float('inf')
        for s in self.stages:
            P = min(P, s.P_in_k[-1])      # capped by this stage's rated input
            P = P * s.eta_at(P)           # what comes out the far side
        return P

    def losses(self):
        """{stage key: array of per-timestep loss [MW]} after solve."""
        return {self.keys[i]: np.array([self.stage_loss(i, t).getValue() for t in self.T]) for i in range(len(self.stages))}


def _val(expr, T):
    """Evaluate a per-timestep expression (or variable) after a solve."""
    def one(e):
        return e.getValue() if hasattr(e, 'getValue') else e.X
    return np.array([one(expr(t)) for t in T])


# --------------------------------------------------------------- kind handlers
# Each handler adds its component's variables and constraints, registers its
# bus injections, and returns a closure that extracts results after the solve.
# Adding a new kind of component = one dataclass + one handler + one registry entry.


def _build_import(m, c, ctx):
    T, dt = ctx['T'], ctx['dt']
    chain = _Chain(m, T, c.stages, c.name)                  # terminal -> bus
    P = m.addVars(T, lb=0.0, ub=c.power, name=f'P_{c.name}')
    m.addConstrs((P[t] == chain.head(t) for t in T), name=f'link_{c.name}')

    ctx['inject'][c.bus].append(lambda t: chain.tail(t))
    price = np.asarray(ctx['sc'].need(c.name, 'price'), dtype=float)
    ctx['obj'].append(gp.quicksum(P[t] * price[t] * dt for t in T))

    def extract():
        return {'P_terminal': np.array([P[t].X for t in T]),
                'P_bus': _val(chain.tail, T), 'losses': chain.losses()}
    return extract


def _build_source(m, c, ctx):
    T = ctx['T']
    avail = np.asarray(ctx['sc'].need(c.name, 'production'), dtype=float)

    if c.exact:
        # Must-run at production: the operating point is known, so walk the chain forward
        # with np.interp instead of handing it to the LP. Forward is exactly solvable
        # (each stage's input is the previous stage's output), unlike the Sink's
        # backward walk, so no approximation is needed here.
        over = avail.max() - c.power
        if over > 1e-9:
            raise ValueError(
                f'{c.name}: exact source must run at its production series, but peak '
                f'{avail.max():.4f} MW exceeds its rating {c.power:.4f} MW.')
        per_stage, P_bus = _static_forward(c.stages, avail)
        ctx['inject'][c.bus].append(lambda t, P=P_bus: P[t])

        def extract():
            return {'P_terminal': avail.copy(), 'P_bus': P_bus.copy(),
                    'losses': per_stage, 'curtailed': np.zeros_like(avail)}
        return extract

    chain = _Chain(m, T, c.stages, c.name)                  # terminal -> bus
    P = m.addVars(T, lb=0.0, ub=c.power, name=f'P_{c.name}')
    m.addConstrs((P[t] == chain.head(t) for t in T), name=f'link_{c.name}')
    m.addConstrs((P[t] <= avail[t] for t in T), name=f'avail_{c.name}')

    ctx['inject'][c.bus].append(lambda t: chain.tail(t))

    def extract():
        return {'P_terminal': np.array([P[t].X for t in T]),
                'P_bus': _val(chain.tail, T), 'losses': chain.losses(),
                'curtailed': avail - np.array([P[t].X for t in T])}
    return extract


def _build_storage(m, c, ctx):
    T, dt, n_t = ctx['T'], ctx['dt'], ctx['n_t']
    dis = _Chain(m, T, c.stages,           f'{c.name}_dis')   # terminal -> bus
    ch  = _Chain(m, T, reversed(c.stages), f'{c.name}_ch')    # bus -> terminal

    P_dis = m.addVars(T, lb=0.0, ub=c.power, name=f'P_{c.name}_dis')   # bus side
    P_ch  = m.addVars(T, lb=0.0, ub=c.power, name=f'P_{c.name}_ch')    # bus side
    m.addConstrs((P_dis[t] == dis.tail(t) for t in T), name=f'link_{c.name}_dis')
    m.addConstrs((P_ch[t]  == ch.head(t)  for t in T), name=f'link_{c.name}_ch')
    for t in T: m.addSOS(GRB.SOS_TYPE1, [P_ch[t], P_dis[t]])

    # SOC limits are fractions of capacity; the variable itself is MWh.
    soc = m.addVars(range(n_t + 1), lb=c.minimum_soc * c.capacity, ub=c.maximum_soc * c.capacity, name=f'SOC_{c.name}')
    
    m.addConstrs((soc[t + 1] == c.eta_self * soc[t] + ch.tail(t) * dt - dis.head(t) * dt for t in T), name=f'soc_{c.name}')
    f_init  = ctx['sc'].need(c.name, 'soc_init')          # fraction of capacity
    f_final = ctx['sc'].opt(c.name, 'soc_final')          #None = free end
    for tag, f in (('soc_init', f_init),) + ((('soc_final', f_final),) if f_final is not None else ()):
        if not c.minimum_soc - 1e-9 <= f <= c.maximum_soc + 1e-9:
            raise ValueError(
                f'{c.name}: {tag} = {f} is a fraction of capacity and must lie within '
                f'[{c.minimum_soc}, {c.maximum_soc}]. For {f} MWh pass {f / c.capacity:.4f}.')
    soc[0].LB = soc[0].UB = f_init * c.capacity
    if f_final is not None: m.addConstr(soc[n_t] == f_final * c.capacity, name=f'soc_terminal_{c.name}')

    ctx['inject'][c.bus].append(lambda t: dis.tail(t) - ch.head(t))
    # ctx['obj'].append(gp.quicksum(5 * (P_ch[t] + P_dis[t]) * dt for t in T))    


    def extract():
        # ----- new change here
        out_MWh = float(np.sum([P_dis[t].X for t in T]) * dt)
        return {'capacity_MWh': c.capacity,
                'throughput_MWh': out_MWh,
                'cycles': out_MWh / c.capacity,
                'P_charge': np.array([P_ch[t].X for t in T]),
                'P_discharge': np.array([P_dis[t].X for t in T]),
                'SOC': np.array([soc[t].X for t in range(n_t + 1)]),
                'losses': {f'{k}_ch': v for k, v in ch.losses().items()}
                        | {f'{k}_dis': v for k, v in dis.losses().items()}}
    return extract


def _build_sink(m, c, ctx):
    T, dt = ctx['T'], ctx['dt']
    demand = np.asarray(np.broadcast_to(
        np.asarray(ctx['sc'].need(c.name, 'demand'), dtype=float), (len(T),)))

    if c.exact:
        # Uncontrollable: the load draws demand[t] whatever the price, so its power
        # is known before the solve and nothing here enters the LP.
        if demand.max() > c.power + 1e-9:
            raise ValueError(
                f'{c.name}: peak demand {demand.max():.4f} MW exceeds its connection '
                f'rating {c.power:.4f} MW.')
        per_stage, P_bus, inputs = _static_backward(c.stages, demand)
        for key, s in zip(_stage_keys(c.stages), c.stages):
            if inputs[key].max() > s.rated + 1e-9:
                raise ValueError(
                    f'{c.name}: stage {s.name} must carry {inputs[key].max():.4f} MW to '
                    f'serve a {demand.max():.4f} MW load, above its rating {s.rated:.4f} '
                    f'MW. Stage ratings are INPUT-referred: rate it at >= load / eta.')
        ctx['inject'][c.bus].append(lambda t, P=P_bus: -P[t])

        def extract():
            return {'P_terminal': demand.copy(), 'P_bus': P_bus.copy(),
                    'demand': demand.copy(), 'losses': per_stage, 'eta': demand / P_bus}
        return extract

    # Controllable: the LP picks the profile. Energy over the horizon is preserved,
    # so the load shifts in time instead of vanishing.
    chain = _Chain(m, T, reversed(c.stages), c.name)        # bus -> terminal
    ub = min(c.power, chain.max_deliverable())
    need_MWh = float(demand.sum()) * dt
    if need_MWh > ub * len(T) * dt + 1e-9:
        raise ValueError(
            f'{c.name}: {need_MWh:.4f} MWh of demand cannot be served at '
            f'{ub:.4f} MW over {len(T)} steps.')

    P = m.addVars(T, lb=0.0, ub=ub, name=f'P_{c.name}')
    m.addConstrs((chain.tail(t) == P[t] for t in T), name=f'serve_{c.name}')
    m.addConstr(gp.quicksum(P[t] for t in T) * dt == need_MWh, name=f'energy_{c.name}')
    ctx['inject'][c.bus].append(lambda t: -chain.head(t))

    def extract():
        served = np.array([P[t].X for t in T])
        bus    = _val(chain.head, T)
        return {'P_terminal': served, 'P_bus': bus, 'demand': demand.copy(),
                'shifted': served - demand, 'losses': chain.losses(),
                'eta': np.divide(served, bus, out=np.ones_like(bus), where=bus > 1e-12)}
    return extract


def _static_backward(stages, P_terminal, iters=100, tol=1e-13):
    """Known terminal (output) power -> bus, inverting each stage exactly.

    P_out = P_in * eta(P_in) is implicit in P_in, so fixed-point iterate
    P_in <- P_out / eta(P_in). eta is close to 1 and slowly varying, so this
    converges in a handful of steps.

    Returns ({stage key: loss array}, bus-side power array, {stage key: input array}).
    """
    P = np.atleast_1d(np.asarray(P_terminal, dtype=float)).astype(float)
    losses, inputs = {}, {}
    for key, s in zip(_stage_keys(stages), stages):      # terminal -> bus
        P_in = P.copy()
        for _ in range(iters):
            nxt = P / np.array([s.eta_at(x) for x in P_in])
            done = np.max(np.abs(nxt - P_in)) < tol
            P_in = nxt
            if done:
                break
        losses[key], inputs[key] = P_in - P, P_in
        P = P_in
    return losses, P, inputs


def _static_forward(stages, P_terminal):
    """Known terminal power -> bus, each stage evaluated at its own input.

    P_terminal may be a scalar or a per-timestep array. Returns
    ({stage key: loss array}, bus-side power array).
    """
    P = np.atleast_1d(np.asarray(P_terminal, dtype=float)).astype(float)
    out = {}
    for key, s in zip(_stage_keys(stages), stages):
        eta   = np.array([s.eta_at(x) for x in P])
        P_out = P * eta
        out[key] = P - P_out
        P = P_out
    return out, P


def _build_link(m, c, ctx):
    T = ctx['T']
    fwd = _Chain(m, T, c.stages,           f'{c.name}_fwd')   # bus -> to_bus
    P_f = m.addVars(T, lb=0.0, ub=c.cap, name=f'P_{c.name}_fwd')
    m.addConstrs((P_f[t] == fwd.head(t) for t in T), name=f'link_{c.name}_fwd')
    ctx['inject'][c.bus].append(lambda t: -fwd.head(t))
    ctx['inject'][c.to_bus].append(lambda t: fwd.tail(t))

    rev = None
    if c.bidirectional:
        rev = _Chain(m, T, reversed(c.stages), f'{c.name}_rev')   # to_bus -> bus
        P_r = m.addVars(T, lb=0.0, ub=c.cap, name=f'P_{c.name}_rev')
        m.addConstrs((P_r[t] == rev.head(t) for t in T), name=f'link_{c.name}_rev')
        ctx['inject'][c.to_bus].append(lambda t: -rev.head(t))
        ctx['inject'][c.bus].append(lambda t: rev.tail(t))

    def extract():
        r = {'P_fwd': np.array([P_f[t].X for t in T]),
             'losses': {f'{k}_fwd': v for k, v in fwd.losses().items()}}
        if rev is not None:
            r['P_rev'] = np.array([P_r[t].X for t in T])
            r['losses'] |= {f'{k}_rev': v for k, v in rev.losses().items()}
        return r
    return extract


HANDLERS = {
    Import:  _build_import,
    Source:  _build_source,
    Storage: _build_storage,
    Sink:    _build_sink,
    Link:    _build_link,
}


# ------------------------------------------------------------------- the builder

def build(spec, scenario=None, name='microgrid', verbose=False):
    """Build the Gurobi model from a plant spec + a run scenario.

    Returns (model, extractors, ctx).
    """
    sc = scenario if scenario is not None else Scenario()
    T = range(sc.n_t)
    m = gp.Model(name)
    m.setParam('OutputFlag', 1 if verbose else 0)

    ctx = {'T': T, 'dt': sc.dt, 'n_t': sc.n_t, 'sc': sc, 'inject': {b: [] for b in spec.buses}, 'obj': []}

    if any(st.sos2 for c in spec.components for st in c.stages):
        m.setParam('PreSOS2BigM', 1.0) # faster solve time with SOS2 constraint

    extractors = {}
    for c in spec.components:
        handler = HANDLERS.get(type(c))
        if handler is None:
            raise TypeError(f'{c.name}: no handler registered for {type(c).__name__}')
        extractors[c.name] = handler(m, c, ctx)

    for b in spec.buses:
        terms = ctx['inject'][b]
        if not terms:
            continue
        m.addConstrs((gp.quicksum(f(t) for f in terms) == 0 for t in T), name=f'balance_{b}')

    m.setObjective(gp.quicksum(ctx['obj']), GRB.MINIMIZE)
    return m, extractors, ctx


STATUS = {
    GRB.OPTIMAL: 'optimal',
    GRB.INFEASIBLE: 'infeasible',
    GRB.UNBOUNDED: 'unbounded',
    GRB.INF_OR_UNBD: 'infeasible_or_unbounded',
    GRB.TIME_LIMIT: 'time_limit',
    GRB.SUBOPTIMAL: 'suboptimal',
}


def build_and_solve(spec, scenario=None, name='microgrid', verbose=False):
    """
    Solve one day for a given spec.

    Returns
    -------
    dict with
      status        : solver termination string
      cost_eur      : objective value
      components    : {name: {...}} per-component arrays (see handlers)
      losses_MWh    : {'component.stage': energy lost [MWh]}
      total_loss_MWh, served_MWh, imported_MWh, efficiency
    """
    sc = scenario if scenario is not None else Scenario()
    m, extractors, ctx = build(spec, sc, name=name, verbose=verbose)
    m.optimize()
    status = STATUS.get(m.Status, f'status_code_{m.Status}')
    if m.Status != GRB.OPTIMAL:
        return {'status': status, 'cost_eur': None, 'components': None, 'modelname': name, 'date': sc.date}

    comps = {n: f() for n, f in extractors.items()}
    dt = ctx['dt']

    losses = {}
    for n, r in comps.items():
        for s, arr in r.get('losses', {}).items():
            losses[f'{n}.{s}'] = float(np.sum(arr) * dt)
    total_loss = float(sum(losses.values()))
    
    demand_supply = sum(float(np.sum(r['P_terminal']) * dt) for c, r in zip(spec.components, comps.values()) if isinstance(c, Sink))
    grid_import = sum(float(np.sum(comps[c.name]['P_terminal']) * dt) for c in spec.components if isinstance(c, Import))
    DER_generation = sum(float(np.sum(comps[c.name]['P_terminal']) * dt) for c in spec.components if isinstance(c, Source))
    
    total_import_and_generation = grid_import + DER_generation
    
    return {
        'modelname': name,
        'date': sc.date,
        'dt': sc.dt,
        'n_t': sc.n_t,
        'status': status,
        'cost_eur': m.ObjVal,
        'components': comps,
        'losses_MWh': losses,
        'total_loss_MWh': total_loss,
        'demand_supply_MWh': demand_supply,
        'grid_import_MWh': grid_import,
        'DER_generation_MWh': DER_generation,
        'efficiency': demand_supply / total_import_and_generation if total_import_and_generation else np.nan,
    }
