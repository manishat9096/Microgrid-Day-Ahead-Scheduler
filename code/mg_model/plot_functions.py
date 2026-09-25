"""
Plots and tables for results from modular_MG_model.build_and_solve.

Tables return a DataFrame and print it only if asked:

    summary_table(results, show = True)
    loss_table(results, show = True)

Plots return (fig, ax)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

FIG_SIZE = (10, 4)
GRID_KW  = dict(axis = 'y', lw = 0.4, alpha = 0.4)

PALETTE = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
           '#937860', '#da8bc3', '#8c8c8c']

COMP_HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
             '#008300', '#4a3aa7', '#e34948']          #hue per component
HATCHES   = ['', '///', '...', 'xxx', '\\']       #texture per model

def _as_list(results):
    """Accept one result, a list, or a dict of them."""
    if isinstance(results, dict) and 'status' in results:
        return [results]
    if isinstance(results, dict):
        return list(results.values())
    return list(results)


def _ok(results):
    """Only the runs that solved."""
    return [r for r in _as_list(results) if r.get('status') == 'optimal']


def _label(r):
    return r.get('modelname') or 'model'


def _colours(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _pivot(results, value):
    """date x modelname table of one scalar field, in first-seen model order."""
    rows = [{'date': r.get('date'), 'model': _label(r), 'value': value(r)}
            for r in _ok(results)]
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df['model']))
    piv = df.pivot_table(index = 'date', columns = 'model', values = 'value')
    return piv.reindex(columns = order)


def _bar(piv, ylabel, title, ax = None):
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize = FIG_SIZE)
    piv.plot(kind = 'bar', ax = ax, width = 0.8,
             color = [_colours(piv.columns)[m] for m in piv.columns])
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc = 'left', fontsize = 11)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if piv.shape[1] > 1:
        ax.legend(title = None, frameon = False, ncol = min(piv.shape[1], 4))
    else:
        ax.get_legend().remove()
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.tight_layout()
    return fig, ax


def _component(r, kind_key):
    """First component carrying a given key, e.g. 'SOC' -> the storage."""
    for name, c in (r.get('components') or {}).items():
        if kind_key in c:
            return name, c
    return None, None


# --------------------------------------------------------------------- plots

def plot_cost(results, ax = None):
    """Daily electricity cost per model."""
    piv = _pivot(results, lambda r: r['cost_eur'])
    return _bar(piv, 'Cost (EUR)', 'Total daily electricity cost', ax)


def plot_grid_import(results, ax = None):
    """Energy drawn from every Import component, metered side."""
    piv = _pivot(results, lambda r: r['grid_import_MWh'])
    return _bar(piv, 'Energy (MWh)', 'Total energy from grid', ax)


def plot_losses(results, ax = None):
    """Total conversion loss across every stage."""
    piv = _pivot(results, lambda r: r['total_loss_MWh'])
    return _bar(piv, 'Losses (MWh)', 'Total conversion losses', ax)


def plot_bess_utilization(results, ax = None):
    """Equivalent full cycles per day (discharged MWh / usable capacity)."""
    def cycles(r):
        _, c = _component(r, 'cycles')
        return np.nan if c is None else c['cycles']
    piv = _pivot(results, cycles)
    return _bar(piv, 'Cycles (MWh out / capacity)', 'BESS utilisation', ax)


def plot_efficiency(results, ax = None):
    """Served energy as a share of everything imported and generated."""
    piv = _pivot(results, lambda r: r['efficiency'] * 100)
    return _bar(piv, 'Efficiency (%)', 'System efficiency', ax)


def _stage_colours(cols):
    """Hue identifies the component, shade the stage within it.

    Stage keys are 'component.stage', so all bess stages come out as shades of one
    hue rather than four unrelated colours. Ten stages never need ten hues.
    """
    comps = list(dict.fromkeys(c.split('.')[0] for c in cols))
    out = {}
    for i, comp in enumerate(comps):
        mine = [c for c in cols if c.split('.')[0] == comp]
        rgb  = np.array(mcolors.to_rgb(COMP_HUES[i % len(COMP_HUES)]))
        for j, c in enumerate(mine):
            f = 0.0 if len(mine) == 1 else 0.5 * j / (len(mine) - 1)
            out[c] = tuple(rgb + (1.0 - rgb) * f)      #lighten toward white
    return out


def plot_loss_breakdown(results, ax = None, stacked = True):
    """Per-stage losses: one group of bars per date, one bar per model.

    Colour is the conversion stage (hue = component, shade = stage within it) and
    hatch is the model, so the models sit side by side at each date and a stage a
    model does not have simply leaves a gap in its stack.

    stacked = False draws each model's total instead of its stage breakdown.
    """
    df = loss_table(results, by_date = True).drop(columns = 'total').fillna(0.0)
    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_arrays(
            [df.index, [''] * len(df)], names = ['model', 'date'])

    models = list(dict.fromkeys(df.index.get_level_values('model')))
    dates  = list(dict.fromkeys(df.index.get_level_values('date')))
    col    = _stage_colours(df.columns)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize = (min(18, max(FIG_SIZE[0], 0.45 * len(dates) * len(models) + 3)), 4.8))

    x, w = np.arange(len(dates), dtype = float), 0.8 / len(models)
    for i, m in enumerate(models):
        off  = (i - (len(models) - 1) / 2) * w
        hat  = HATCHES[i % len(HATCHES)]
        bot  = np.zeros(len(dates))
        segs = df.columns if stacked else []
        for stage in segs:
            v = np.array([df.loc[(m, d), stage] if (m, d) in df.index else 0.0 for d in dates])
            ax.bar(x + off, v, bottom = bot, width = w * 0.88, color = col[stage],
                   hatch = hat, lw = 0.0, zorder = 3)
            bot += v
        if not stacked:
            bot = np.array([df.loc[(m, d)].sum() if (m, d) in df.index else 0.0 for d in dates])
            ax.bar(x + off, bot, width = w * 0.88, color = COMP_HUES[i % len(COMP_HUES)],
                   hatch = hat, lw = 0.0, zorder = 3)

    pad = max(0.0, (3 - len(dates)) * 0.6)      #keep bars narrow when there are few dates
    ax.set_xlim(-0.5 - pad, len(dates) - 0.5 + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dates])
    ax.set_xlabel('')
    ax.set_ylabel('Loss (MWh)')
    ax.set_title('Losses by conversion stage' if stacked else 'Total conversion losses',
                 loc = 'left', fontsize = 11, pad = 30)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation = 30, ha = 'right')

    comps = list(dict.fromkeys(c.split('.')[0] for c in df.columns))
    comp_h = [mpatches.Patch(facecolor = COMP_HUES[i % len(COMP_HUES)], label = c)
              for i, c in enumerate(comps)] if stacked else []
    model_h = [mpatches.Patch(facecolor = '#dddddd' if stacked else COMP_HUES[i % len(COMP_HUES)],
                              hatch = HATCHES[i % len(HATCHES)], label = m)
               for i, m in enumerate(models)]
    handles = comp_h + model_h
    leg = ax.legend(handles = handles, frameon = False, fontsize = 9, ncol = len(handles),
                    bbox_to_anchor = (0.0, 1.02), loc = 'lower left',
                    handlelength = 1.4, columnspacing = 1.4)
    ax.add_artist(leg)
    fig.subplots_adjust(right = 0.97, top = 0.84, bottom = 0.25)
    return fig, ax


def _kind(c):
    """Infer a component's kind from the keys its extractor returned."""
    if 'SOC' in c:       return 'storage'
    if 'demand' in c:    return 'sink'
    if 'curtailed' in c: return 'source'
    return 'import'


def _series(r):
    """[(label, MW array)] for every power flow in one result."""
    out, n = [], r['n_t']
    for name, c in (r.get('components') or {}).items():
        k = _kind(c)
        if k == 'storage':
            out += [(f'{name} charge', c['P_charge']), (f'{name} discharge', c['P_discharge'])]
        elif k in ('import', 'source'):
            out.append((name, c['P_terminal']))
        else:
            out.append((name, np.atleast_1d(c['P_terminal'])[:n]))
    return out


def plot_dispatch(results, index = None, figsize = (12, 5), flows = None):
    """Power flows and SOC over one day.

    Colour identifies the flow, linestyle the model, so two architectures overlay
    without the colours moving. `flows` optionally restricts which series to draw
    (by label). `index` is a DatetimeIndex of length n_t for the x axis.
    """
    rs     = _ok(results)
    labels = list(dict.fromkeys(l for r in rs for l, _ in _series(r)))
    if flows is not None:
        labels = [l for l in labels if l in flows]
    col    = _colours(labels)                      # colour follows the flow
    styles = ['-', '--', '-.', ':']

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize = figsize, sharex = True,
        gridspec_kw = {'height_ratios': [2.2, 1], 'hspace': 0.08})

    for r, ls in zip(rs, styles * 4):
        t = index if index is not None else np.arange(r['n_t'])
        for label, y in _series(r):
            if label in col:
                ax1.plot(t, y, color = col[label], ls = ls, lw = 1.6)
        _, st = _component(r, 'SOC')
        if st is not None:
            ax2.plot(t, st['SOC'][:r['n_t']], color = '#555', ls = ls, lw = 1.8)

    ax1.set_ylabel('Power (MW)')
    ax1.set_ylim(bottom = 0)
    ax2.set_ylabel('SOC (MWh)')
    for a in (ax1, ax2):
        a.grid(**GRID_KW)
        a.set_axisbelow(True)
        a.spines[['top', 'right']].set_visible(False)

    if index is not None:
        ax2.xaxis.set_major_locator(mdates.HourLocator(byhour = range(0, 25, 3)))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.set_xlabel('Time of day')
    else:
        ax2.set_xlabel('Timestep')

    handles  = [mlines.Line2D([0], [0], color = col[l], lw = 2, label = l) for l in labels]
    handles += [mlines.Line2D([0], [0], color = '#555', ls = ls, lw = 1.6, label = _label(r))
                for r, ls in zip(rs, styles * 4)]
    ax1.legend(handles = handles, frameon = False, fontsize = 8,
               ncol = 1, bbox_to_anchor = (1.01, 1.0), loc = 'upper left')
    date = rs[0].get('date')
    ax1.set_title(f'Day-ahead dispatch{"  -  " + str(date) if date else ""}',
                  loc = 'left', fontsize = 11)
    fig.subplots_adjust(right = 0.80)   # room for the legend, no tight_layout warning
    return fig, (ax1, ax2)


# -------------------------------------------------------------------- tables

def summary_table(results, show = False, sort = None):
    """One row per run: cost, efficiency, stage count and the energy totals.

    n_stages counts the conversion stages that actually reported a loss, so a
    model that drops a converter shows a smaller count.
    """
    rows = []
    for r in _as_list(results):
        if r.get('status') != 'optimal':
            rows.append({'model': _label(r), 'date': r.get('date'),
                         'status': r.get('status')})
            continue
        _, st = _component(r, 'cycles')
        rows.append({
            'model'             : _label(r),
            'date'              : r.get('date'),
            'status'            : r['status'],
            'cost_eur'          : r['cost_eur'],
            'efficiency_pct'    : r['efficiency'] * 100,
            'n_stages'          : len(r['losses_MWh']),
            'total_loss_MWh'    : r['total_loss_MWh'],
            'demand_supply_MWh' : r['demand_supply_MWh'],
            'grid_import_MWh'   : r['grid_import_MWh'],
            'DER_generation_MWh': r['DER_generation_MWh'],
            'bess_cycles'       : np.nan if st is None else st['cycles'],
        })
    df = pd.DataFrame(rows)
    if sort:
        df = df.sort_values(sort)
    if show:
        print(df.to_string(index = False, float_format = lambda x: f'{x:,.4f}'))
    return df


def loss_table(results, show = False, by_date = None):
    """Model x stage losses in MWh. A stage a model does not have is NaN.

    Stage keys are 'component.stage', so two models are only pooled into the same
    column when they name the component and the stage the same way.
    """
    rs = _ok(results)
    dates = {r.get('date') for r in rs}
    if by_date is None:
        by_date = len(dates) > 1 and dates != {None}

    idx = [(_label(r), r.get('date')) for r in rs] if by_date else [_label(r) for r in rs]
    df = pd.DataFrame([r['losses_MWh'] for r in rs], index = idx)
    if by_date:
        df.index = pd.MultiIndex.from_tuples(df.index, names = ['model', 'date'])
    else:
        df.index.name = 'model'
    df = df.reindex(columns = sorted(df.columns))
    df['total'] = df.sum(axis = 1)
    if show:
        print(df.to_string(float_format = lambda x: f'{x:,.4f}', na_rep = '-'))
    return df


def stage_share_table(results, show = False):
    """loss_table as a percentage of each model's own total."""
    df = loss_table(results)
    out = df.drop(columns = 'total').div(df['total'], axis = 0) * 100
    if show:
        print(out.to_string(float_format = lambda x: f'{x:,.1f}', na_rep = '-'))
    return out


def compare_table(results, baseline = None, show = False):
    """Every model against one baseline: deltas in cost, loss and efficiency."""
    df = summary_table(results).set_index('model')
    base = baseline or df.index[0]
    b = df.loc[base]
    out = pd.DataFrame({
        'cost_eur'      : df['cost_eur'],
        'dcost_eur'     : df['cost_eur'] - b['cost_eur'],
        'dcost_pct'     : (df['cost_eur'] - b['cost_eur']) / b['cost_eur'] * 100,
        'total_loss_MWh': df['total_loss_MWh'],
        'dloss_MWh'     : df['total_loss_MWh'] - b['total_loss_MWh'],
        'efficiency_pct': df['efficiency_pct'],
        'deff_pp'       : df['efficiency_pct'] - b['efficiency_pct'],
    })
    if show:
        print(f'baseline: {base}')
        print(out.to_string(float_format = lambda x: f'{x:,.4f}'))
    return out
