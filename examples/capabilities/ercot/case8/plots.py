# Copyright (C) 2017-2022 Battelle Memorial Institute
# file: plots.py

# usage 'python plots metrics_root'
import os
import sys
import tesp_support.api.process_gld as gp
import tesp_support.api.process_houses as hp
import tesp_support.original.process_agents as ap
import tesp_support.api.process_voltages as vp

rootname = sys.argv[1]

gp.process_gld (rootname)
hp.process_houses (rootname)
vp.process_voltages (rootname)

exists = os.path.isfile ('auction_' + rootname + '_metrics.json')
if exists:
  ap.process_agents (rootname)


