#!/usr/bin/env python
# coding: utf-8
import os 
import sys

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

for datatype in SETTINGS.data.keys():
    file_data = get_file_classes(datatype)
    files_found = len(file_data)
    print("\n{} data: {} files.".format(datatype, files_found))
    if (files_found > 0):
        for kit_label in SETTINGS.label["kit_label"]:
            data_count = 0
            for fd in file_data:
                if (kit_label in fd["labels"]["kit_label"]):
                    data_count += 1
            print("Found {} : {}".format(data_count, kit_label))

input("Press enter to continue...")
exit()


