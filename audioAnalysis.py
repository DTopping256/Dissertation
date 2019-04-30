#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import statistics as stats
import soundfile as sf
import sys
import os

# Allows me to import my modules
sys.path.append('./modules')

# The python definition of f(x) for problem 1.
from audio_utils import *

data = read_data("raw", batch_size=5)

reduced_d0 = reduce_sample_rate(data[1], 3000)

plt.plot(reduced_d0.data)
plt.title("Amplitude over time ({}-{})".format(reduced_d0.labels["kit_label"][0], reduced_d0.labels["tech_label"][0]))
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (arb. units)")
plt.show()

bd_local_energy = local_energy_func(reduced_d0.data, 100)
plt.plot(bd_local_energy)
plt.title("Bass drum local energy")
plt.show()

bd_ebn = energy_based_novelty(reduced_d0.data, 100)
plt.plot(bd_ebn)
plt.title("Bass drum energy based novelty")
plt.show()

bd_ebn_sd = stats.stdev(bd_ebn.tolist())
bd_ebn_tf = threshold_func(bd_ebn, 300, 2*bd_ebn_sd, 5)
plt.plot(bd_ebn, color="red")
plt.plot(bd_ebn_tf, color="black")
plt.legend(["Δ log energy", "Threshold function"])
plt.title("EBN with threshold ({}-{})".format(reduced_d0.labels["kit_label"][0], reduced_d0.labels["tech_label"][0]))
plt.ylabel("Energy (arb. units)")
plt.xlabel("Time (samples)")
plt.show()

print("Onsets:", onset_detection(bd_ebn, bd_ebn_tf, trim=100))