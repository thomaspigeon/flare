import numpy as np
import time
import os

# Load AgI data.
AgI_location = "https://zenodo.org/record/3688843/files/AgI_data.zip?download=1"
wget_return = os.system("wget %s" % AgI_location)
# For Macs:
if wget_return != 0:
    os.system("curl %s -o AgI_data.zip?download=1" % AgI_location)
os.system("unzip AgI_data.zip?download=1")

