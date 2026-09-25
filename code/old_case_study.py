import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'                   

import numpy as np
import pandas as pd


from mg_model import modular_MG_model as mod
from specifications import SPEC
from architechture import AC_Arch, DC_Arch


# Load Data 
spot_df = pd.read_csv(DATA / 'spot_df.csv', parse_dates = ['HourUTC'])
price_15min = spot_df.set_index('HourUTC')['SpotPriceEUR'].resample('15min').ffill()

pv_raw = pd.read_csv(DATA / 'pv_raw.csv', index_col = 'time', parse_dates = ['time'])
pv_15min = (pv_raw['P'] / 1e6).resample('15min').interpolate('linear')

# Day-ahead scheduling test one day
date = '2025-06-01'
price_day = price_15min[date].values
pv_day    = pv_15min[date].values
load_day  = np.full(SPEC.N_T, SPEC.LOAD_MW) 
idx        = price_15min[date].index

oneday = mod.Scenario(n_t = SPEC.N_T, dt = SPEC.DT, data = {'grid': {'price': price_day},
                                            'PV'  : {'production': pv_day},
                                            'bess': {'soc_init': 0.5, 'soc_final': 0.5},
                                            'load': {'demand': load_day}})


t0 = time.perf_counter()
result = mod.build_and_solve(AC_Arch, oneday, verbose = False)
runtime = time.perf_counter() - t0   # build + solve + extract, seconds
print(f'model run time {runtime:.3f} s')

def tree(d, pre = ''):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f'{pre}{k}/')
            tree(v, pre + '  ')
        else:
            a = np.asarray(v)
            what = f'array{a.shape}  {a.min():.4g} .. {a.max():.4g}' if a.ndim else f'{v}'
            print(f'{pre}{k:<14s} {what}')

tree(result)          # or tree(c) for just the components


# Results
c = result['components']

print(f"\n{date}   {result['status']}   cost {result['cost_eur']:.2f} EUR   " f"efficiency {result['efficiency'] * 100:.2f} %")
print(f"grid {result['grid_import_MWh']:.3f} MWh   pv {result['DER_generation_MWh']:.3f} MWh   " f"load {result['demand_supply_MWh']:.3f} MWh   losses {result['total_loss_MWh']:.3f} MWh")

print('\nlosses by stage [MWh]')
for k, v in sorted(result['losses_MWh'].items(), key = lambda kv: -kv[1]):
    print(f'  {k:24s} {v:7.4f}   {v / result["total_loss_MWh"] * 100:5.1f} %')

soc = c['bess']['SOC']
cycled = c['bess']['P_discharge'].sum() * SPEC.DT
print(f"\nbess   soc {soc.min():.2f} - {soc.max():.2f} MWh   " f"in {c['bess']['P_charge'].sum() * SPEC.DT:.3f} MWh   out {cycled:.3f} MWh   " f"{cycled / SPEC.BESS_CAP_MWH:.2f} cycles")
print(f"pv     used {c['PV']['P_terminal'].sum() * SPEC.DT:.3f} MWh   " f"curtailed {c['PV']['curtailed'].sum() * SPEC.DT:.3f} MWh")


dispatch = pd.DataFrame({'price': price_day, 'grid': c['grid']['P_terminal'], 'pv': c['PV']['P_terminal'], 'charge': c['bess']['P_charge'], 'discharge': c['bess']['P_discharge'], 'soc': soc[:-1]}, index = idx)
hourly = dispatch.resample('h').agg({'price': 'mean', 'grid': 'mean', 'pv': 'mean', 'charge': 'mean', 'discharge': 'mean', 'soc': 'last'})
print('\nhourly dispatch [EUR/MWh, MW, MWh]')
print(hourly.round(3).to_string())


# AC vs DC over the same random days
import matplotlib.pyplot as plt
from mg_model import plot_functions as pf

rng = np.random.default_rng(0)
all_days = pd.date_range('2025-01-02', '2025-09-29', freq = 'D')
days = pd.DatetimeIndex(rng.choice(all_days, size = 20, replace = False)).sort_values().strftime('%Y-%m-%d')

results = []
for arch, name in [(AC_Arch, 'AC'), (DC_Arch, 'DC')]:
    for d in days:
        sc = mod.Scenario(n_t = SPEC.N_T, dt = SPEC.DT, date = d,
                          data = {'grid': {'price': price_15min[d].values}, 'PV': {'production': pv_15min[d].values}, 'bess': {'soc_init': 0.5}, 'load': {'demand': load_day}})
        results.append(mod.build_and_solve(arch, sc, name = name, verbose = False))

print('\nsummary'); pf.summary_table(results, show = True)
print('\nlosses by stage [MWh]'); pf.loss_table(results, show = True)
print('\nstage share of own total [%]'); pf.stage_share_table(results, show = True)

for f in (pf.plot_cost, pf.plot_grid_import, pf.plot_losses, pf.plot_efficiency, pf.plot_bess_utilization, pf.plot_loss_breakdown):
    f(results)

d = days[9]
pf.plot_dispatch([r for r in results if r['date'] == d], index = price_15min[d].index)
plt.show()
