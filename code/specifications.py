from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class ModelSpec:
    PV_RATED_MW: float = 1.83
    BESS_POW_MW: float = 5.0
    BESS_CAP_MWH: float = 9.0
    SOC_MIN: float = 0.1
    SOC_MAX: float = 0.9
    LOAD_RATING: float = 4.0
    LOAD_MW: float = 3.53
    GRID_CAP_MW: float = 10.0
    DT: float = 0.25
    N_T: int = 96

@dataclass(frozen=True)
class ConverterEfficiencyCurves:
    Inverter_for_PV: ClassVar[tuple[list[float], list[float]]] = ([.05, .10, .20, .30, .50, .75, 1.00], [.890, .932, .962, .972, .978, .977, .974])
    Inverter_for_BESS: ClassVar[tuple[list[float], list[float]]] = ([.10, .20, .30, .50, .75, 1.00], [.905, .928, .953, .974, .973, .970])
    Converter_for_PV: ClassVar[tuple[list[float], list[float]]] = ([.05, .10, .20, .30, .50, .75, 1.00], [.905, .938, .963, .973, .979, .978, .976])
    Converter_for_BESS: ClassVar[tuple[list[float], list[float]]] = ([.05, .10, .20, .30, .50, .75, 1.00], [.895, .922, .951, .964, .974, .973, .970])
    Transformer: ClassVar[tuple[list[float], list[float]]] = ([.10, .20, .30, .50, .75, 1.00], [.950, .967, .978, .988, .988, .986])
    Inverter_for_Load: ClassVar[tuple[list[float], list[float]]] = ([.05, .10, .20, .30, .50, .75, 1.00], [.870, .900, .930, .948, .961, .960, .957])
    
SPEC = ModelSpec()
EFFICIENCYCURVE = ConverterEfficiencyCurves()