#!/usr/bin/env python
# coding: utf-8

import sys
from functools import reduce
import constraint as c 

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

def hit_breakdown():
    breakdown = {hl: [] for hl in SETTINGS.label["hit_label"]}
    for hl in SETTINGS.label["hit_label"]:
        for kl in SETTINGS.hierarchy[hl].keys():
            for tl in SETTINGS.hierarchy[hl][kl]:
                breakdown[hl].append("-".join([kl, tl]))
    return breakdown

def unique_kit_class(*ts):
    current_kt = ts[0].split("-")[0]
    for t in ts[1:]:
        kt = t.split("-")[0]
        if kt == current_kt:
            return False
    return True

def kit_combinations():
    hit_combos = SETTINGS.data["multiclassed"]["hit_combinations"]
    breakdown = hit_breakdown()
    kit_combos = []
    # For each hit combination do constraint programming to find all combinations of kit_label - tech_label hits
    for hc in hit_combos:
        prob = c.Problem()
        var_i = 0
        for ht in hc:
            prob.addVariable(str(var_i), breakdown[ht])
            var_i += 1
        # Kit type contraint (they must be unique)
        prob.addConstraint(unique_kit_class)
        # Remove permutations of hits
        kit_combo_subset = list(reduce(lambda acc, x: [*acc, set(x.values())] if set(x.values()) not in acc else acc, prob.getSolutions(), []))
        # Check elements of subset aren't already in kit_combos list
        kit_combo_subset = list(filter(lambda x: x not in kit_combos, kit_combo_subset))
        # Add to kit_combos
        kit_combos.extend(kit_combo_subset)
    return kit_combos