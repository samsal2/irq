#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from itertools import starmap
from typing import Union, List, Callable, Tuple
from pandas import DataFrame, Series
from scipy.optimize import minimize, root
from scipy.integrate import solve_ivp

# https://doi.org/10.1016/j.fbp.2009.06.003
# https://doi.org/10.1016/j.renene.2019.10.060

DEFAULT_R = 8.314 # J / mol K
DEFAULT_STANDARD_T = 293.15 # K
SUBSTANCE_INDEX = [
    "Glycerol", 
    "Acetic Acid", 
    "Water",
    "Triacetin", 
    "Diacetin", 
    "Monoacetin"
]

substance_data = DataFrame(
    [[221.18, -669.6e3,  92.0938],
     [159.8,  -483.52e3, 60.0520],
     [75.38,  -285.83e3, 60.0520],
     [389.0,  -1330.8e3, 218.2039],
     [340.98, -1120.7e3, 176.1672],
     [291.36, -903.53e3, 134.1305]],
    index=SUBSTANCE_INDEX,
    columns=["Cp (J / mol K)", "Hf (J / mol)", "MW (g / mol)"]
)

reaction_data = DataFrame(
    [[6.9, 3.1e4, 190, 3.3e4],
     [6.8, 3.1e4, 220, 3.8e4],
     [2.4, 3.4e4, 200, 4.3e4]],
    index=[1, 2, 3],
    columns=["A", "Ea", "A-1", "Ea-1"]
)


def arrhenius(A: np.float64, Ea: np.float64, T: np.float64, R: np.float64=DEFAULT_R) -> np.float64:
    return A * np.exp(-Ea / (R * T))


def _concentrations_as_series(c: List[np.float64]) -> Series:
    return Series(c, index=SUBSTANCE_INDEX)


def r1(c: List[np.float64], T: np.float64) -> np.float64:
    k = arrhenius(reaction_data.loc[1]["A"], reaction_data.loc[1]["Ea"], T)
    kinv = arrhenius(reaction_data.loc[1]["A-1"], reaction_data.loc[1]["Ea-1"], T)

    s = _concentrations_as_series(c)
    return -k * s["Glycerol"] * s["Acetic Acid"] + kinv * s["Monoacetin"] * s["Water"]


def r2(c: List[np.float64], T: np.float64) -> np.float64:
    k = arrhenius(reaction_data.loc[2]["A"], reaction_data.loc[2]["Ea"], T)
    kinv = arrhenius(reaction_data.loc[2]["A-1"], reaction_data.loc[2]["Ea-1"], T)
    
    s = _concentrations_as_series(c)
    return -k * s["Monoacetin"] * s["Acetic Acid"] + kinv * s["Diacetin"] * s["Water"]


def r3(c: List[np.float64], T: np.float64) -> np.float64:
    k = arrhenius(reaction_data.loc[3]["A"], reaction_data.loc[3]["Ea"], T)
    kinv = arrhenius(reaction_data.loc[3]["A-1"], reaction_data.loc[3]["Ea-1"], T)
    
    s = _concentrations_as_series(c)
    return -k * s["Diacetin"] * s["Acetic Acid"] + kinv * s["Triacetin"] * s["Water"]

reaction_data["r"] = [r1, r2, r3]

# storing functions on a dataframe doesn't sound like a good idea, who cares
substance_data["netr"] = [
        lambda c, T: r1(c, T),
        lambda c, T: r1(c, T) + r2(c, T) + r3(c, T),
        lambda c, T: -r1(c, T) - r2(c, T) - r3(c, T),
        lambda c, T: -r3(c, T),
        lambda c, T: -r2(c, T) + r3(c, T),
        lambda c, T: -r1(c, T) + r2(c, T)
]


def hstd(hfr: List[Tuple[int, np.float64]], hfp: List[Tuple[int, np.float64]]) -> np.float64:
    return np.sum(list(starmap(np.multiply, hfp))) - np.sum(list(starmap(np.multiply, hfr)))


hf = substance_data["Hf (J / mol)"]


reaction_data["H˚rxn (J / mol)"] = [
    hstd([(1, hf["Glycerol"]), (1, hf["Acetic Acid"])], [(1, hf["Monoacetin"]), (1, hf["Water"])]),
    hstd([(1, hf["Monoacetin"]), (1, hf["Acetic Acid"])], [(1, hf["Diacetin"]), (1, hf["Water"])]),
    hstd([(1, hf["Diacetin"]), (1, hf["Acetic Acid"])], [(1, hf["Triacetin"]), (1, hf["Water"])])
]


class CSTRExitStreamSolver:
    def __init__(self, **kwargs):
        self._concentration_cache = None
        self.s0 = _concentrations_as_series(kwargs["c0"])

        
    def _create_objective_function(self, T: np.float64, tau: np.float64) -> Callable[[List[np.float64]], List[np.float64]]:
    
        def objective_function(c: List[np.float64]):
            s = _concentrations_as_series(c)
            r = substance_data["netr"]
            return np.array([tau - (self.s0[key] - s[key]) / -r[key](c, T) for key in self.s0.index])
        
        return objective_function
        
        
    def _estimate_concentrations(self, T: np.float64) -> List[np.float64]:
        if not self._concentration_cache is None:
            return self._concentration_cache
        
        def f(t: np.float64, c: List[np.float64]) -> List[np.float64]:
            return np.array([r(c, T) for r in substance_data["netr"]])
                
        return solve_ivp(f, [0, 500], self.s0.to_numpy()).y.transpose()[-1]
    
        
    def solve(self, T: np.float64, tau: np.float64) -> List[np.float64]:
        result = root(self._create_objective_function(T, tau), self._estimate_concentrations(T), method="lm")
        
        if not result.success:
            print(f"no solution for T: {T}, tau: {tau}")
            raise RuntimeError(f"Couldn't converge to a solution T: {T}, tau: {tau}\n{result}")
  
        print(f"found solution for T: {T}, tau: {tau}, x: {result.x}")
        return self._set_and_return_concentration_cache(result.x)

    
    def pop_concentration_cache(self) -> List[np.float64]:
        tmp = self._concentration_cache
        self._concentration_cache = None
        return tmp
        
    def _set_and_return_concentration_cache(self, c: List[np.float64]) -> List[np.float64]:
        self._concentration_cache = c
        return self._concentration_cache


cstr_default_exit_stream_solver = CSTRExitStreamSolver(c0=[0.1, 0.7, 0.0, 0.0, 0.0, 0.0])


@np.vectorize
def cstr_triacetin_yield(T: np.float64, tau: np.float64, cess=cstr_default_exit_stream_solver) -> np.float64:
    s = _concentrations_as_series(cess.solve(T, tau))
    return s["Triacetin"] / (cess.s0["Glycerol"] - s["Glycerol"])


T, tau = np.linspace(350, 495, 10), np.linspace(1000, 12000, 10)
T, tau = np.meshgrid(T, tau)
Y = cstr_triacetin_yield(T, tau)


ax = plt.axes(projection='3d')
ax.plot_surface(T, tau, Y)
plt.show()


# triacetin_yield(590, 12000)





