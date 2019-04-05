# Adapted from Marcin Możejko's callback post
# Ref: https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
import time
import json
import os
from keras.callbacks import Callback

# Logs the start time, end time and epoch durations in ticks and saves them to a json file at the end.
class Time_Callback(Callback):
  def __init__(self, log_dir):
    self.log_dir = log_dir
    self.state = {"start_time": time.time(), "end_time": None, "epoch_durations": []}
    super(Callback, self).__init__()

  def on_epoch_begin(self, batch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, batch, logs={}):
    self.state["epoch_durations"].append(time.time() - self.epoch_time_start)

  def on_train_end(self, logs={}):
    self.state.update({"end_time": time.time()})
    fp = os.path.join(self.log_dir, "time_log.json")
    with open(fp, "wt") as log_file:
      json.dump(self.state, log_file)