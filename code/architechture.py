# building single AC, single DC, and AC-DC Hybrid model here

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

from mg_model import components
from specifications import SPEC, EFFICIENCYCURVE


# build all components
pv_dcac = components.Stage('pv_dcac', SPEC.PV_RATED_MW, *EFFICIENCYCURVE.Inverter_for_PV,)# sos2=True)
pv_dcdc = components.Stage('pv_dcdc', SPEC.PV_RATED_MW, *EFFICIENCYCURVE.Converter_for_PV,)# sos2=True)
b_dcac  = components.Stage('bess_dcac', SPEC.BESS_POW_MW, *EFFICIENCYCURVE.Inverter_for_BESS,)# sos2=True)
b_dcdc  = components.Stage('bess_dcdc', SPEC.BESS_POW_MW, *EFFICIENCYCURVE.Converter_for_BESS,)# sos2=True)
g_dcac  = components.Stage('grid_dcac', SPEC.GRID_CAP_MW, *EFFICIENCYCURVE.Inverter_for_BESS,)# sos2=True)
trafo   = components.Stage('trafo', SPEC.GRID_CAP_MW, *EFFICIENCYCURVE.Transformer,)# sos2=True)
l_acdc  = components.Stage('load_acdc', SPEC.LOAD_RATING, *EFFICIENCYCURVE.Inverter_for_Load,)# sos2=True)
l_dcdc  = components.Stage('load_dcdc', SPEC.LOAD_RATING, *EFFICIENCYCURVE.Converter_for_BESS,)# sos2=True)

# AC Microgrid
buses = ['ac_bus']

# component order (Terminal --> Bus)
PCC = components.Import(name = 'grid', bus = 'ac_bus', stages = [trafo], power = SPEC.GRID_CAP_MW)
PV = components.Source(name = 'PV', bus = 'ac_bus', stages = [pv_dcdc, pv_dcac], power = SPEC.PV_RATED_MW)
BESS = components.Storage(name = 'bess', bus = 'ac_bus', stages = [b_dcdc, b_dcac],  power = SPEC.BESS_POW_MW, capacity = SPEC.BESS_CAP_MWH, 
                                                                    minimum_soc = SPEC.SOC_MIN, maximum_soc= SPEC.SOC_MAX)
DCLoad = components.Sink(name = 'load', bus = 'ac_bus', stages = [l_dcdc, l_acdc], power = SPEC.LOAD_RATING, exact = True)

AC_Arch = components.SystemSpec(buses = buses,  components = [PCC, PV, BESS, DCLoad],) # Build system using SystemSpec


# DC Microgrid
buses = ['dc_bus']

# component order (Terminal --> Bus)
PCC = components.Import(name = 'grid', bus = 'dc_bus', stages = [g_dcac], power = SPEC.GRID_CAP_MW)
PV = components.Source(name = 'PV', bus = 'dc_bus', stages = [pv_dcdc], power = SPEC.PV_RATED_MW)
BESS = components.Storage(name = 'bess', bus = 'dc_bus', stages = [b_dcdc],  power = SPEC.BESS_POW_MW, capacity = SPEC.BESS_CAP_MWH, 
                                                                    minimum_soc = SPEC.SOC_MIN, maximum_soc= SPEC.SOC_MAX)
DCLoad = components.Sink(name = 'load', bus = 'dc_bus', stages = [l_dcdc], power = SPEC.LOAD_RATING, exact = True)

DC_Arch = components.SystemSpec(buses = buses,  components = [PCC, PV, BESS, DCLoad],) # Build system using SystemSpec

# Hybrid 2 Bus AC-DC MG
# buses = ['ac_bus', 'dc_bus']


# AC Microgrid with PV (No curtailment)
buses = ['ac_bus']

# component order (Terminal --> Bus)
PCC = components.Import(name = 'grid', bus = 'ac_bus', stages = [trafo], power = SPEC.GRID_CAP_MW)
PV = components.Source(name = 'PV', bus = 'ac_bus', stages = [pv_dcdc, pv_dcac], power = SPEC.PV_RATED_MW, exact = True)
BESS = components.Storage(name = 'bess', bus = 'ac_bus', stages = [b_dcdc, b_dcac],  power = SPEC.BESS_POW_MW, capacity = SPEC.BESS_CAP_MWH, minimum_soc = SPEC.SOC_MIN, maximum_soc= SPEC.SOC_MAX)
DCLoad = components.Sink(name = 'load', bus = 'ac_bus', stages = [l_dcdc, l_acdc], power = SPEC.LOAD_RATING, exact = False)

Custom_Arch1 = components.SystemSpec(buses = buses,  components = [PCC, PV, BESS, DCLoad],) # Build system using SystemSpec


# AC Microgrid with 2 PV (No curtailment) and 1 DC Load (Fixed)
buses = ['ac_bus']

# component order (Terminal --> Bus)
PCC = components.Import(name = 'grid', bus = 'ac_bus', stages = [trafo], power = SPEC.GRID_CAP_MW)
PV1 = components.Source(name = 'PV1', bus = 'ac_bus', stages = [pv_dcdc, pv_dcac], power = SPEC.PV_RATED_MW, exact = True)
PV2 = components.Source(name = 'PV2', bus = 'ac_bus', stages = [pv_dcdc, pv_dcac], power = SPEC.PV_RATED_MW, exact = False)
BESS = components.Storage(name = 'bess', bus = 'ac_bus', stages = [b_dcdc, b_dcac],  power = SPEC.BESS_POW_MW, capacity = SPEC.BESS_CAP_MWH, minimum_soc = SPEC.SOC_MIN, maximum_soc= SPEC.SOC_MAX)
DCLoad = components.Sink(name = 'load', bus = 'ac_bus', stages = [l_dcdc, l_acdc], power = SPEC.LOAD_RATING, exact = True)

Custom_Arch2 = components.SystemSpec(buses = buses,  components = [PCC, PV, BESS, DCLoad],) # Build system using SystemSpec


