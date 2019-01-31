#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

def create_subdirs(kit_name):
    data_info = SETTINGS.data["raw"]
    root_path = data_info["path"]
    for i in SETTINGS.label["hit_label"]:
        kit = [s for s in SETTINGS.label["kit_label"] if s != "bass_drum"]
        if (i == "beater"):
            kit = ["bass_drum"]
        for j in kit:
            tech = ["normal"]
            if (j == "hi_hat"):
                tech = SETTINGS.label["tech_label"]
            for k in tech:
                path = os.path.join(root_path, kit_name, i, j, k)
                if (not os.path.exists(path)):
                    try:
                        os.makedirs(path)
                    except OSError:
                        print("Couldn't make new path: ", path)
                        return False
    print("Make directory structure at: ", root_path, "under", kit_name)
    return True

parser = argparse.ArgumentParser(description="Generate subdirectories for a new set of raw data")
parser.add_argument("-kit_name")
args = parser.parse_args(sys.argv[1:])
create_subdirs(**vars(args))
exit()
