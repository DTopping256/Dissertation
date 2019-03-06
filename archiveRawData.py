#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import os
import shutil
import sys
import time
import zipfile

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *


# In[14]:


def archiveData(datatype="raw", verbose=False):
    if verbose:
        print("Attempting to archive", datatype, "data.")
    if (SETTINGS is None):
        if verbose:
            print("\tRead failed, since settings not found.")
        return False
    if (datatype not in SETTINGS.data.keys()):
        if verbose:
            print("\t", datatype, "is not a valid datatype.")
        return False
    data_info = SETTINGS.data[datatype]
    root_path = os.path.join(os.getcwd(), data_info["path"])
    try:
        currentfiles = os.listdir(os.getcwd())
        # Filter non datatype.zip files
        currentfiles = list(map(lambda filename: int(filename[len(datatype)+1:-15]), filter(lambda filename: filename[:len(datatype)] == datatype and filename[-4:] == ".zip", currentfiles)))
        currentfiles.sort(reverse=True)
        uid = 0
        if (len(currentfiles) > 0):
            uid = currentfiles[0]+1  
        with zipfile.ZipFile(datatype+"_"+str(uid)+"_"+str(datetime.datetime.now())[:10]+".zip", mode="x") as archive:
            for path, dirs, files in os.walk(root_path):
                for file in files:
                    filepath = os.path.join(path, file)
                    archive.write(filepath, filepath[len(root_path)+1:])
    except:
        print("\tError (archive): ", sys.exc_info())
        return False
    try:
        for path, dirs, files in os.walk(root_path):
            for directory in dirs:
                shutil.rmtree(os.path.relpath(os.path.join(path, directory)))
    except:
        print("\tError (cleaning): ", sys.exc_info())
        return False
    if verbose:
        print("\tSuccesfully archived {} data and cleaned files.".format(datatype))
    return True


# In[15]:


archiveData(verbose=True)


# In[18]:


def restoreData(datatype="raw", archive_filepath=None, verbose=False):
    method = "most recent" if archive_filepath is None else "specified"
    if verbose:
        print("Attempting to restore", datatype, "data.")
    if (SETTINGS is None):
        if verbose:
            print("\tRead failed, since settings not found.")
        return False
    if (datatype not in SETTINGS.data.keys()):
        if verbose:
            print("\t", datatype, "is not a valid datatype.")
        return False
    data_info = SETTINGS.data[datatype]
    target_path = os.path.join(os.getcwd(), data_info["path"])
    try:
        if (archive_filepath is None):
            currentfiles = os.listdir(os.getcwd())
            # Filter non datatype.zip files
            currentfiles = list(map(lambda filename: int(filename[len(datatype)+1:-15]), filter(lambda filename: filename[:len(datatype)] == datatype and filename[-4:] == ".zip", currentfiles)))
            currentfiles.sort(reverse=True)
            if (len(currentfiles) > 0):
                uid = currentfiles[0] 
                archive_filepath = next(fn for fn in os.listdir(os.getcwd()) if datatype+"_"+str(uid) in fn)
            else:
                if verbose:
                    print("\tNo recognised", datatype, "archives in cwd.")
                return False
        with zipfile.ZipFile(archive_filepath, mode="r") as archive:
            archive.extractall(path=target_path)
    except:
        print("\tError (restore archive): ", sys.exc_info())
        return False
    if verbose:
        print("\tSuccesfully restored", method, datatype, "data.")
    return True


# In[20]:


#restoreData(verbose=True)


# In[ ]:




