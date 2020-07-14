# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:40:05 2020

@author: lance
"""

# test commit hooks

import pandas as pd

# From: https://worldpopulationreview.com/states/
POP_CSV = "population.csv"
POP_URL = "https://worldpopulationreview.com/states"

def get_population_by_state()-> dict:
    data: pd.DataFrame = pd.read_csv(POP_CSV)
    states = list(data.State.values)
    pop = list(data.Pop.values)
    return dict(zip(states, pop))
