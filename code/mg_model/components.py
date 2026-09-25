"""
Data classes describing a microgrid: what a converter is, what a component is, and
how they are wired together. Pure data + a little validation, no solver in sight.

**Nothing here changes between runs.** Ratings, efficiency curves, SOC limits and the
wiring are properties of the plant, so a SystemSpec is built once and reused for every
day, price series and scenario. Everything that varies run to run -- prices, PV
production, initial/final SOC, horizon and timestep -- is scenario data and lives
with the model (see `Scenario` in modular_MG_model.py), not here.

`Stage` is one conversion stage (a rated power and an efficiency table). A component
carries an ordered list of them, always written terminal -> bus. `SystemSpec` names the
buses and holds the components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------- efficiency data

@dataclass
class Stage:
    """One conversion stage: a rated power and a tabulated efficiency curve.

    loading/eta are the table from the datasheet. A zero-power breakpoint is
    prepended automatically so the lambda weights can represent P = 0 (without it
    sum(lam) == 1 would force the stage to carry at least its smallest tabulated
    power at every timestep).

    `rated` is INPUT-referred: a stage can never deliver its own rating, so a
    load-side converter serving P must be rated at least P / eta.

    sos2 tightens only this stage. Without it the lambda weights may mix
    non-adjacent breakpoints and ride the lower convex envelope of the loss curve,
    understating loss; with it the two nonzero weights must be neighbours, which is
    the true piecewise-linear interpolation. It turns the LP into a MIP, so set it
    on the stages whose curves actually need it. It has no effect on a stage that
    never reaches the LP (an exact Sink or must-run Source).
    """
    name: str
    rated: float
    loading: Sequence[float]
    eta: Sequence[float]
    sos2: bool = False               # enforce adjacent breakpoints for THIS stage

    def __post_init__(self):
        l = np.asarray(self.loading, dtype=float)
        e = np.asarray(self.eta, dtype=float)
        if l.size != e.size:
            raise ValueError(f'{self.name}: loading and eta differ in length')
        if l[0] > 0.0:
            l = np.insert(l, 0, 0.0)
            e = np.insert(e, 0, e[0])
        self.loading, self.eta = l, e
        self.P_in_k  = l * self.rated        # breakpoint input powers  [MW]
        self.P_out_k = self.P_in_k * e       # breakpoint output powers [MW]

    @property
    def K(self):
        return range(len(self.P_in_k))

    def eta_at(self, P):
        """Exact np.interp efficiency at a fixed operating point."""
        return float(np.interp(np.clip(P / self.rated, 0.0, 1.0), self.loading, self.eta))


# ------------------------------------------------------------------- system spec

@dataclass(kw_only=True)
class Component:
    """Base: a terminal attached to one bus by a chain of stages (terminal -> bus).

    `stages` is always a LIST of Stage objects, never a count. Any length works:
    two stages, one stage, or `stages = []` for a direct lossless connection to the
    bus. Dropping a converter is deleting a list element.
    """
    name: str
    bus: str
    stages: list = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.stages, Stage):
            self.stages = [self.stages]                 # a bare Stage is a chain of one
        if not isinstance(self.stages, (list, tuple)):
            raise TypeError(
                f'{self.name}: stages must be a list of Stage objects, got '
                f'{self.stages!r}. Use [] for a direct connection, [s] for one stage.')
        bad = [s for s in self.stages if not isinstance(s, Stage)]
        if bad:
            raise TypeError(f'{self.name}: stages contains non-Stage entries: {bad!r}')
        self.stages = list(self.stages)


@dataclass(kw_only=True)
class Import(Component):
    """Grid connection. Terminal-side power is metered and priced.
    """
    power: float


@dataclass(kw_only=True)
class Source(Component):
    """Generation. The output series is scenario data: Scenario.data[name]['production'].

    exact=False (default) is curtailable generation such as PV: the LP picks any
    output up to production[t], so the operating point is a decision and the conversion
    chain goes into the LP as lambda blocks.

    exact=True is must-run generation at a fixed output, such as a DG set held at a
    setpoint: output IS production[t], no curtailment. Because the terminal power is then
    known before the solve, the chain is evaluated with np.interp instead -- each
    stage at its own true input -- and contributes a known injection to the bus at
    no cost to the LP. Stage ratings are checked, since production[t] must physically fit
    through the chain.
    """
    power: float
    exact: bool = False              # False = curtailable, True = must-run


@dataclass(kw_only=True)
class Storage(Component):
    """Battery. Bidirectional; SOC tracks terminal-side power.

    minimum_soc/maximum_soc are FRACTIONS of capacity
    """
    power: float
    capacity: float
    minimum_soc: float
    maximum_soc: float
    eta_self: float = 1.0            # battery efficiency

    def __post_init__(self):
        super().__post_init__()
        for tag, f in (('minimum_soc', self.minimum_soc), ('maximum_soc', self.maximum_soc)):
            if not 0.0 <= f <= 1.0:
                raise ValueError(
                    f'{self.name}: {tag} = {f} must be a fraction of capacity in [0, 1], '
                    f'not MWh (capacity = {self.capacity}).')
        if self.minimum_soc >= self.maximum_soc:
            raise ValueError(f'{self.name}: minimum_soc must be below maximum_soc')


@dataclass(kw_only=True)
class Sink(Component):
    """Load. `power` is the connection RATING; the demand is scenario data:
    Scenario.data[name]['demand'], MW per step (a scalar is broadcast).

    Mirrors Source. exact says whether the load is a given or a decision.

    exact=True (default) is an uncontrollable load: it draws demand[t], full stop.
    The terminal power is known before the solve, so the chain never enters the LP --
    each stage is inverted exactly for the input that delivers its known output.

    exact=False is a controllable load: the LP decides the profile, capped per step
    by the connection rating, and the conversion chain goes in as lambda blocks.
    Total energy over the horizon is held at sum(demand)*dt, so the load is shifted
    in time rather than shed. Without that the LP would simply serve nothing.
    """
    power: float                     # connection rating [MW]
    exact: bool = True               # True = uncontrollable, False = LP decides


@dataclass(kw_only=True)
class Link(Component):
    """Converter tying two buses together. `bus` is the from-side, `to_bus` the to-side."""
    to_bus: str
    cap: float
    bidirectional: bool = True


@dataclass
class SystemSpec:
    """The plant: which buses exist and what hangs off them.

    Horizon and timestep are NOT here -- they belong to a run, so they live on
    Scenario. One SystemSpec is built once and solved against many Scenarios.
    """
    buses: list
    components: list

    def __post_init__(self):
        seen = set()
        for c in self.components:
            if c.name in seen:
                raise ValueError(f'duplicate component name: {c.name}')
            seen.add(c.name)
            for b in (c.bus, getattr(c, 'to_bus', c.bus)):
                if b not in self.buses:
                    raise ValueError(f'{c.name}: unknown bus {b!r}')

    def get(self, name):
        return next(c for c in self.components if c.name == name)
