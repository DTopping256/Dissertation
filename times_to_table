#!/usr/bin/env python
# coding: utf-8

import os
import json

time_data = {}
for log in os.listdir(os.path.join(os.getcwd(), "logs")):
    with open(os.path.join(os.getcwd(), "logs", log, "time_log.json")) as time_log:
        time_data.update({log: json.load(time_log)})

get_optim = lambda x: x.split("_")[0].split("-")[-1]

optimiser_times = {get_optim(k): v for k, v in time_data.items() if k[:22] == "ModelB-(1D)-OneHot-BCE"}

model_times = {k: v for k, v in time_data.items() if get_optim(k) == "ADAM"}

def make_time_table(data):
    print("Durations per epoch (mins)")
    print(" & ".join(["Epoch", *list(map(str, range(1, 11)))]))
    for k, v in data.items():
        print(" & ".join([k, map(lambda t: str(round(t/60, 1)), v[epoch_durations])]))

if __name__ == "__main__":
    print("Optimiser times")
    make_time_table(optimiser_times)
    print("Model times")
    make_time_table(model_times)

