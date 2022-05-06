
# Copyright (C) 2018-2020 Battelle Memorial Institute
# file: feederGenerator.py
"""Replaces ZIP loads with houses, and optional storage and solar generation.

As this module populates the feeder backbone wiht houses and DER, it uses
the Networkx package to perform graph-based capacity analysis, upgrading
fuses, transformers and lines to serve the expected load. Transformers have
a margin of 20% to avoid overloads, while fuses have a margin of 150% to
avoid overloads. These can be changed by editing tables and variables in the
source file.

There are two kinds of house populating methods implemented:

    * :Feeders with Service Transformers: This case applies to the full PNNL taxonomy feeders. Do not specify the *taxchoice* argument to *populate_feeder*. Each service transformer receiving houses will have a short service drop and a small number of houses attached.
    * :Feeders without Service Transformers: This applies to the reduced-order ERCOT feeders. To invoke this mode, specify the *taxchoice* argument to *populate_feeder*. Each primary load to receive houses will have a large service transformer, large service drop and large number of houses attached.

References:
    `GridAPPS-D Feeder Models <https://github.com/GRIDAPPSD/Powergrid-Models>`_

Public Functions:
    :populate_feeder: processes one GridLAB-D input file

Todo:
    * Verify the level zero mobile home thermal integrity properties; these were copied from the MATLAB feeder generator

"""
import sys;
import re;
import os.path;
import networkx as nx;
import numpy as np;
import argparse
import math;
import json;
import tesp_support.helpers as helpers;

global ConfigDict
global c_p_frac
extra_billing_meters = set()

#***************************************************************************************************
#***************************************************************************************************

def write_node_house_configs (fp, xfkva, xfkvll, xfkvln, phs, want_inverter=False):
  """Writes transformers, inverter settings for GridLAB-D houses at a primary load point.

  An aggregated single-phase triplex or three-phase quadriplex line configuration is also
  written, based on estimating enough parallel 1/0 AA to supply xfkva load.
  This function should only be called once for each combination of xfkva and phs to use,
  and it should be called before write_node_houses.

  Args:
      fp (file): Previously opened text file for writing; the caller closes it.
      xfkva (float): the total transformer size to serve expected load; make this big enough to avoid overloads
      xfkvll (float): line-to-line voltage [kV] on the primary. The secondary voltage will be 208 three-phase
      xfkvln (float): line-to-neutral voltage [kV] on the primary. The secondary voltage will be 120/240 for split secondary
      phs (str): either 'ABC' for three-phase, or concatenation of 'A', 'B', and/or 'C' with 'S' for single-phase to triplex
      want_inverter (boolean): True to write the IEEE 1547-2018 smarter inverter function setpoints
  """
  if want_inverter:
    print ('#define INVERTER_MODE=CONSTANT_PF', file=fp)
    print ('//#define INVERTER_MODE=VOLT_VAR', file=fp)
    print ('//#define INVERTER_MODE=VOLT_WATT', file=fp)
    print ('// default IEEE 1547-2018 settings for Category B', file=fp)
    print ('#define INV_V1=0.92', file=fp)
    print ('#define INV_V2=0.98', file=fp)
    print ('#define INV_V3=1.02', file=fp)
    print ('#define INV_V4=1.08', file=fp)
    print ('#define INV_Q1=0.44', file=fp)
    print ('#define INV_Q2=0.00', file=fp)
    print ('#define INV_Q3=0.00', file=fp)
    print ('#define INV_Q4=-0.44', file=fp)
    print ('#define INV_VIN=200.0', file=fp)
    print ('#define INV_IIN=32.5', file=fp)
    print ('#define INV_VVLOCKOUT=300.0', file=fp)
    print ('#define INV_VW_V1=1.05 // 1.05833', file=fp)
    print ('#define INV_VW_V2=1.10', file=fp)
    print ('#define INV_VW_P1=1.0', file=fp)
    print ('#define INV_VW_P2=0.0', file=fp)
  if 'S' in phs:
    for secphs in phs.rstrip('S'):
      xfkey = 'XF{:s}_{:d}'.format (secphs, int(xfkva))
      write_xfmr_config (xfkey, secphs + 'S', kvat=xfkva, vnom=None, vsec=120.0, install_type='PADMOUNT', vprimll=None, vprimln=1000.0*xfkvln, op=fp)
    write_kersting_triplex (fp, xfkva)
  else:
    xfkey = 'XF3_{:d}'.format (int(xfkva))
    write_xfmr_config (xfkey, phs, kvat=xfkva, vnom=None, vsec=208.0, install_type='PADMOUNT', vprimll=1000.0*xfkvll, vprimln=None, op=fp)
    write_kersting_quadriplex (fp, xfkva)

#***************************************************************************************************
#***************************************************************************************************

def write_kersting_quadriplex (fp, kva):
  """Writes a quadriplex_line_configuration based on 1/0 AA example from Kersting's book

  The conductor capacity is 202 amps, so the number of triplex in parallel will be kva/sqrt(3)/0.208/202
  """
  key = 'quad_cfg_{:d}'.format (int(kva))
  amps = kva / math.sqrt(3.0) / 0.208
  npar = math.ceil (amps / 202.0)
  apar = 202.0 * npar
  scale = 5280.0 / 100.0 / npar  # for impedance per mile of parallel circuits
  r11 = 0.0268 * scale
  x11 = 0.0160 * scale
  r12 = 0.0080 * scale
  x12 = 0.0103 * scale
  r13 = 0.0085 * scale
  x13 = 0.0095 * scale
  r22 = 0.0258 * scale
  x22 = 0.0176 * scale
  print ('object line_configuration {{ // {:d} 1/0 AA in parallel'.format (int(npar)), file=fp)
  print ('  name {:s};'.format (key), file=fp)
  print ('  z11 {:.4f}+{:.4f}j;'.format (r11, x11), file=fp)
  print ('  z12 {:.4f}+{:.4f}j;'.format (r12, x12), file=fp)
  print ('  z13 {:.4f}+{:.4f}j;'.format (r13, x13), file=fp)
  print ('  z21 {:.4f}+{:.4f}j;'.format (r12, x12), file=fp)
  print ('  z22 {:.4f}+{:.4f}j;'.format (r22, x22), file=fp)
  print ('  z23 {:.4f}+{:.4f}j;'.format (r12, x12), file=fp)
  print ('  z31 {:.4f}+{:.4f}j;'.format (r13, x13), file=fp)
  print ('  z32 {:.4f}+{:.4f}j;'.format (r12, x12), file=fp)
  print ('  z33 {:.4f}+{:.4f}j;'.format (r11, x11), file=fp)
  print ('  rating.summer.continuous {:.1f};'.format (apar), file=fp)
  print ('  rating.summer.emergency {:.1f};'.format (apar), file=fp)
  print ('  rating.winter.continuous {:.1f};'.format (apar), file=fp)
  print ('  rating.winter.emergency {:.1f};'.format (apar), file=fp)
  print ('}', file=fp)

#***************************************************************************************************
#***************************************************************************************************

def write_kersting_triplex (fp, kva):
  """Writes a triplex_line_configuration based on 1/0 AA example from Kersting's book

  The conductor capacity is 202 amps, so the number of triplex in parallel will be kva/0.12/202
  """
  key = 'tpx_cfg_{:d}'.format (int(kva))
  amps = kva / 0.12
  npar = math.ceil (amps / 202.0)
  apar = 202.0 * npar
  scale = 5280.0 / 100.0 / npar  # for impedance per mile of parallel circuits
  r11 = 0.0271 * scale
  x11 = 0.0146 * scale
  r12 = 0.0087 * scale
  x12 = 0.0081 * scale
  print ('object triplex_line_configuration {{ // {:d} 1/0 AA in parallel'.format (int(npar)), file=fp)
  print ('  name {:s};'.format (key), file=fp)
  print ('  z11 {:.4f}+{:.4f}j;'.format (r11, x11), file=fp)
  print ('  z12 {:.4f}+{:.4f}j;'.format (r12, x12), file=fp)
  print ('  z21 {:.4f}+{:.4f}j;'.format (r12, x12), file=fp)
  print ('  z22 {:.4f}+{:.4f}j;'.format (r11, x11), file=fp)
  print ('  rating.summer.continuous {:.1f};'.format (apar), file=fp)
  print ('  rating.summer.emergency {:.1f};'.format (apar), file=fp)
  print ('  rating.winter.continuous {:.1f};'.format (apar), file=fp)
  print ('  rating.winter.emergency {:.1f};'.format (apar), file=fp)
  print ('}', file=fp)

#***************************************************************************************************
#***************************************************************************************************

def union_of_phases(phs1, phs2):
    """Collect all phases on both sides of a connection

    Args:
        phs1 (str): first phasing
        phs2 (str): second phasing

    Returns:
        str: union of phs1 and phs2
    """
    phs = ''
    if 'A' in phs1 or 'A' in phs2:
        phs += 'A'
    if 'B' in phs1 or 'B' in phs2:
        phs += 'B'
    if 'C' in phs1 or 'C' in phs2:
        phs += 'C'
    if 'S' in phs1 or 'S' in phs2:
        phs += 'S'
    return phs

#***************************************************************************************************
#***************************************************************************************************

def accumulate_load_kva(data):
    """Add up the total kva in a load-bearing object instance

    Considers constant_power_A/B/C/1/2/12 and power_1/2/12 attributes

    Args:
        data (dict): dictionary of data for a selected GridLAB-D instance
    """
    kva = 0.0
    if 'constant_power_A' in data:
        kva += parse_kva(data['constant_power_A'])
    if 'constant_power_B' in data:
        kva += parse_kva(data['constant_power_B'])
    if 'constant_power_C' in data:
        kva += parse_kva(data['constant_power_C'])
    if 'constant_power_1' in data:
        kva += parse_kva(data['constant_power_1'])
    if 'constant_power_2' in data:
        kva += parse_kva(data['constant_power_2'])
    if 'constant_power_12' in data:
        kva += parse_kva(data['constant_power_12'])
    if 'power_1' in data:
        kva += parse_kva(data['power_1'])
    if 'power_2' in data:
        kva += parse_kva(data['power_2'])
    if 'power_12' in data:
        kva += parse_kva(data['power_12'])
    return kva

#***************************************************************************************************
#***************************************************************************************************

def log_model(model, h):
    """Prints the whole parsed model for debugging

    Args:
        model (dict): parsed GridLAB-D model
        h (dict): object ID hash
    """
    for t in model:
        print(t+':')
        for o in model[t]:
            print('\t'+o+':')
            for p in model[t][o]:
                if ':' in model[t][o][p]:
                    print('\t\t'+p+'\t-->\t'+h[model[t][o][p]])
                else:
                    print('\t\t'+p+'\t-->\t'+model[t][o][p])
#***************************************************************************************************
#***************************************************************************************************

def randomize_commercial_skew():
  sk = ConfigDict['commercial_skew_std']['value'] * np.random.randn ()
  if sk < -ConfigDict['commercial_skew_max']['value']:
    sk = -ConfigDict['commercial_skew_max']['value']
  elif sk > ConfigDict['commercial_skew_max']['value']:
    sk = ConfigDict['commercial_skew_max']['value']
  return sk

#***************************************************************************************************
#***************************************************************************************************

def write_config_class (model, h, t, op):
    """Write a GridLAB-D configuration (i.e. not a link or node) class

    Args:
        model (dict): the parsed GridLAB-D model
        h (dict): the object ID hash
        t (str): the GridLAB-D class
        op (file): an open GridLAB-D input file
    """
    if t in model:
        for o in model[t]:
#            print('object ' + t + ':' + o + ' {', file=op)
            print('object ' + t + ' {', file=op)
            print('  name ' + o + ';', file=op)
            for p in model[t][o]:
                if ':' in model[t][o][p]:
                    print ('  ' + p + ' ' + h[model[t][o][p]] + ';', file=op)
                else:
                    print ('  ' + p + ' ' + model[t][o][p] + ';', file=op)
            print('}', file=op)

#***************************************************************************************************
#***************************************************************************************************


def is_edge_class(s):
    """Identify switch, fuse, recloser, regulator, transformer, overhead_line, underground_line and triplex_line instances

    Edge class is networkx terminology. In GridLAB-D, edge classes are called links.

    Args:
        s (str): the GridLAB-D class name

    Returns:
        Boolean: True if an edge class, False otherwise
    """
    if s == 'switch':
        return True
    if s == 'fuse':
        return True
    if s == 'recloser':
        return True
    if s == 'regulator':
        return True
    if s == 'transformer':
        return True
    if s == 'overhead_line':
        return True
    if s == 'underground_line':
        return True
    if s == 'triplex_line':
        return True
    return False

#***************************************************************************************************
#***************************************************************************************************

def is_node_class(s):
    """Identify node, load, meter, triplex_node or triplex_meter instances

    Args:
        s (str): the GridLAB-D class name

    Returns:
        Boolean: True if a node class, False otherwise
    """
    if s == 'node':
        return True
    if s == 'load':
        return True
    if s == 'meter':
        return True
    if s == 'triplex_node':
        return True
    if s == 'triplex_meter':
        return True
    return False

#***************************************************************************************************
#***************************************************************************************************

def parse_kva_old(arg):
    """Parse the kVA magnitude from GridLAB-D P+jQ volt-amperes in rectangular form

    DEPRECATED

    Args:
        cplx (str): the GridLAB-D P+jQ value

    Returns:
        float: the parsed kva value
    """
    tok = arg.strip('; MWVAKdrij')
    nsign = nexp = ndot = 0
    for i in range(len(tok)):
        if (tok[i] == '+') or (tok[i] == '-'):
            nsign += 1
        elif (tok[i] == 'e') or (tok[i] == 'E'):
            nexp += 1
        elif tok[i] == '.':
            ndot += 1
        if nsign == 2 and nexp == 0:
            kpos = i
            break
        if nsign == 3:
            kpos = i
            break

    vals = [tok[:kpos],tok[kpos:]]
#    print(arg,vals)

    vals = [float(v) for v in vals]

    if 'd' in arg:
        vals[1] *= (math.pi / 180.0)
        p = vals[0] * math.cos(vals[1])
        q = vals[0] * math.sin(vals[1])
    elif 'r' in arg:
        p = vals[0] * math.cos(vals[1])
        q = vals[0] * math.sin(vals[1])
    else:
        p = vals[0]
        q = vals[1]

    if 'KVA' in arg:
        p *= 1.0
        q *= 1.0
    elif 'MVA' in arg:
        p *= 1000.0
        q *= 1000.0
    else:  # VA
        p /= 1000.0
        q /= 1000.0

    return math.sqrt (p*p + q*q)

#***************************************************************************************************
#***************************************************************************************************

def parse_kva(cplx): # this drops the sign of p and q
    """Parse the kVA magnitude from GridLAB-D P+jQ volt-amperes in rectangular form

    Args:
        cplx (str): the GridLAB-D P+jQ value

    Returns:
        float: the parsed kva value
    """
    toks = list(filter(None,re.split('[\+j-]',cplx)))
    p = float(toks[0])
    q = float(toks[1])
    return 0.001 * math.sqrt(p*p + q*q)

#***************************************************************************************************
#***************************************************************************************************

def selectResidentialBuilding(rgnTable,prob):
    """Writes volt-var and volt-watt settings for solar inverters

    Args:
        op (file): an open GridLAB-D input file
    """
    row = 0
    total = 0
    for row in range(len(rgnTable)):
        for col in range(len(rgnTable[row])):
            total += rgnTable[row][col]
            if total >= prob:
                return row, col
    row = len(rgnTable) - 1
    col = len(rgnTable[row]) - 1
    return row, col


#***************************************************************************************************
#***************************************************************************************************

def buildingTypeLabel (rgn, bldg, ti):
    """Formatted name of region, building type name and thermal integrity level

    Args:
        rgn (int): region number 1..5
        bldg (int): 0 for single-family, 1 for apartment, 2 for mobile home
        ti (int): thermal integrity level, 0..6 for single-family, only 0..2 valid for apartment or mobile home
    """
    return ConfigDict['rgnName']['value'][rgn-1] + ': ' + ConfigDict['bldgTypeName']['value'][bldg] + ': TI Level ' + str (ti+1)

#***************************************************************************************************
#***************************************************************************************************

def Find3PhaseXfmr (kva):
    """Select a standard 3-phase transformer size, with data

    Standard sizes are 30, 45, 75, 112.5, 150, 225, 300, 500, 750, 1000, 1500,
    2000, 2500, 3750, 5000, 7500 or 10000 kVA

    Args:
        kva (float): the minimum transformer rating

    Returns:
        [float,float,float,float,float]: the kva, %r, %x, %no-load loss, %magnetizing current
    """
    for row in ConfigDict['three_phase']['value']:
        if row[0] >= kva:
            return row[0], 0.01 * row[1], 0.01 * row[2], 0.01 * row[3], 0.01 * row[4]
    return Find3PhaseXfmrKva(kva),0.01,0.08,0.005,0.01


#***************************************************************************************************
#***************************************************************************************************

def Find1PhaseXfmr (kva):
    """Select a standard 1-phase transformer size, with data

    Standard sizes are 5, 10, 15, 25, 37.5, 50, 75, 100, 167, 250, 333 or 500 kVA

    Args:
        kva (float): the minimum transformer rating

    Returns:
        [float,float,float,float,float]: the kva, %r, %x, %no-load loss, %magnetizing current
    """
    for row in ConfigDict['single_phase']['value']:
        if row[0] >= kva:
            return row[0], 0.01 * row[1], 0.01 * row[2], 0.01 * row[3], 0.01 * row[4]
    return Find1PhaseXfmrKva(kva),0.01,0.06,0.005,0.01

#***************************************************************************************************
#***************************************************************************************************
def Find3PhaseXfmrKva (kva):
    """Select a standard 3-phase transformer size, with some margin

    Standard sizes are 30, 45, 75, 112.5, 150, 225, 300, 500, 750, 1000, 1500,
    2000, 2500, 3750, 5000, 7500 or 10000 kVA

    Args:
        kva (float): the minimum transformer rating

    Returns:
        float: the kva size, or 0 if none found
    """
    #kva *= xfmrMargin
    kva *= ConfigDict['xmfr']['xfmrMargin']['value']
    for row in ConfigDict['three_phase']['value']:
        if row[0] >= kva:
            return row[0]
    n10 = int ((kva + 5000.0) / 10000.0)
    return 500.0 * n10

#***************************************************************************************************
#***************************************************************************************************


def Find1PhaseXfmrKva (kva):
    """Select a standard 1-phase transformer size, with some margin

    Standard sizes are 5, 10, 15, 25, 37.5, 50, 75, 100, 167, 250, 333 or 500 kVA

    Args:
        kva (float): the minimum transformer rating

    Returns:
        float: the kva size, or 0 if none found
    """
    #kva *= xfmrMargin
    kva *= ConfigDict['xmfr']['xfmrMargin']['value']
    for row in ConfigDict['single_phase']['value']:
        if row[0] >= kva:
            return row[0]
    n500 = int ((kva + 250.0) / 500.0)
    return 500.0 * n500

#***************************************************************************************************
#***************************************************************************************************

def checkResidentialBuildingTable():
    """Verify that the regional building parameter histograms sum to one
    """

    for tbl in range(len(ConfigDict['rgnThermalPct']['value'])):
        total = 0
        for row in range(len(ConfigDict['rgnThermalPct']['value'][tbl])):
            for col in range(len(ConfigDict['rgnThermalPct']['value'][tbl][row])):
                total += ConfigDict['rgnThermalPct'][tbl]['value'][row][col]
        print (ConfigDict['rgnName']['value'][tbl],'rgnThermalPct sums to', '{:.4f}'.format(total))
    for tbl in range(len(ConfigDict['bldgCoolingSetpoints']['value'])):
        total = 0
        for row in range(len(ConfigDict['bldgCoolingSetpoints']['value'][tbl])):
            total += ConfigDict['bldgCoolingSetpoints']['value'][tbl][row][0]
        print ('bldgCoolingSetpoints', tbl, 'histogram sums to', '{:.4f}'.format(total))
    for tbl in range(len(ConfigDict['bldgHeatingSetpoints']['value'])):
        total = 0
        for row in range(len(ConfigDict['bldgHeatingSetpoints']['value'][tbl])):
            total += ConfigDict['bldgHeatingSetpoints']['value'][tbl][row][0]
        print ('bldgHeatingSetpoints', tbl, 'histogram sums to', '{:.4f}'.format(total))
    for bldg in range(3):
        binZeroReserve = ConfigDict['bldgCoolingSetpoints']['value'][bldg][0][0]
        binZeroMargin = ConfigDict['bldgHeatingSetpoints']['value'][bldg][0][0] - binZeroReserve
        if binZeroMargin < 0.0:
            binZeroMargin = 0.0
#        print (bldg, binZeroReserve, binZeroMargin)
        for cBin in range(1, 6):
            denom = binZeroMargin
            for hBin in range(1, ConfigDict['allowedHeatingBins']['value'][cBin]):
                denom += ConfigDict['bldgHeatingSetpoints']['value'][bldg][hBin][0]
            ConfigDict['conditionalHeatingBinProb']['value'][bldg][cBin][0] = binZeroMargin / denom
            for hBin in range(1, ConfigDict['allowedHeatingBins']['value'][cBin]):
                ConfigDict['conditionalHeatingBinProb']['value'][bldg][cBin][hBin] = ConfigDict['bldgHeatingSetpoints']['value'][bldg][hBin][0] / denom
#    print ('conditionalHeatingBinProb', ConfigDict['conditionalHeatingBinProb']['value'])

#***************************************************************************************************
#***************************************************************************************************

def selectThermalProperties(bldgIdx, tiIdx):
    """Retrieve the building thermal properties for a given type and integrity level

    Args:
        bldgIdx (int): 0 for single-family, 1 for apartment, 2 for mobile home
        tiIdx (int): 0..6 for single-family, 0..2 for apartment or mobile home
    """
    if bldgIdx == 0:
        tiProps = ConfigDict['singleFamilyProperties']['value'][tiIdx]
    elif bldgIdx == 1:
        tiProps = ConfigDict['apartmentProperties']['value'][tiIdx]
    else:
        tiProps = ConfigDict['mobileHomeProperties']['value'][tiIdx]
    return tiProps


#***************************************************************************************************
#***************************************************************************************************

def FindFuseLimit (amps):
    """ Find a Fuse size that's unlikely to melt during power flow

    Will choose a fuse size of 40, 65, 100 or 200 Amps.
    If that's not large enough, will choose a recloser size
    of 280, 400, 560, 630 or 800 Amps. If that's not large
    enough, will choose a breaker size of 600 (skipped), 1200
    or 2000 Amps. If that's not large enough, will choose 999999.

    Args:
        amps (float): the maximum load current expected; some margin will be added

    Returns:
        float: the GridLAB-D fuse size to insert
    """
    #amps *= fuseMargin
    amps *= ConfigDict['fuseMargin']['value']
    for row in ConfigDict['standard_fuses']['value']:
        if row >= amps:
            return row
    for row in ConfigDict['standard_reclosers']['value']:
        if row >= amps:
            return row
    for row in ConfigDict['standard_breakers']['value']:
        if row >= amps:
            return row
    return 999999

#***************************************************************************************************
#***************************************************************************************************

def selectSetpointBins (bldg, rand):
    """Randomly choose a histogram row from the cooling and heating setpoints
    The random number for the heating setpoint row is generated internally.
    Args:
        bldg (int): 0 for single-family, 1 for apartment, 2 for mobile home
        rand (float): random number [0..1] for the cooling setpoint row
    """
    global ConfigDict
    cBin = hBin = 0
    total = 0
    tbl = ConfigDict['bldgCoolingSetpoints']['value'][bldg]
    for row in range(len(tbl)):
        total += tbl[row][0]
        if total >= rand:
            cBin = row
            break
    tbl = ConfigDict['conditionalHeatingBinProb']['value'][bldg][cBin]
    rand_heat = np.random.uniform (0, 1)
    total = 0
    for col in range(len(tbl)):
        total += tbl[col]
        if total >= rand_heat:
            hBin = col
            break
    ConfigDict['cooling_bins']['value'][bldg][cBin] -= 1
    ConfigDict['heating_bins']['value'][bldg][hBin] -= 1
    return ConfigDict['bldgCoolingSetpoints']['value'][bldg][cBin], ConfigDict['bldgHeatingSetpoints']['value'][bldg][hBin]

#***************************************************************************************************
#***************************************************************************************************
#fgconfig: path and name of the file that is to be used as the configuration json for loading
#ConfigDict dictionary
def initialize_config_dict(fgconfig):
    global ConfigDict
    global c_p_frac
    if fgconfig is not None:
        ConfigDict = {}
        with open(fgconfig,'r') as fgfile:
            confile = fgfile.read()
            ConfigDict = json.loads(confile)
            fgfile.close()
        tval2 = ConfigDict['feedergenerator']['constants']
        ConfigDict = tval2
        cval1 = ConfigDict['c_z_frac']['value']
        cval2 = ConfigDict['c_i_frac']['value']
        #c_p_frac = 1.0 - ConfigDict['c_z_frac'] - ConfigDict['c_i_frac']
        c_p_frac = 1.0 - cval1 - cval2
#       fgfile.close()

#***************************************************************************************************
#***************************************************************************************************
#***************************************************************************************************
#***************************************************************************************************
def write_solar_inv_settings (op):
    """Writes volt-var and volt-watt settings for solar inverters

    Args:
        op (file): an open GridLAB-D input file
    """
    #print ('    four_quadrant_control_mode ${' + name_prefix + 'INVERTER_MODE};', file=op)
    print ('    four_quadrant_control_mode ${' + ConfigDict['name_prefix']['value'] + 'INVERTER_MODE};', file=op)
    print ('    V_base ${INV_VBASE};', file=op)
    print ('    V1 ${INV_V1};', file=op)
    print ('    Q1 ${INV_Q1};', file=op)
    print ('    V2 ${INV_V2};', file=op)
    print ('    Q2 ${INV_Q2};', file=op)
    print ('    V3 ${INV_V3};', file=op)
    print ('    Q3 ${INV_Q3};', file=op)
    print ('    V4 ${INV_V4};', file=op)
    print ('    Q4 ${INV_Q4};', file=op)
    print ('    V_In ${INV_VIN};', file=op)
    print ('    I_In ${INV_IIN};', file=op)
    print ('    volt_var_control_lockout ${INV_VVLOCKOUT};', file=op)
    print ('    VW_V1 ${INV_VW_V1};', file=op)
    print ('    VW_V2 ${INV_VW_V2};', file=op)
    print ('    VW_P1 ${INV_VW_P1};', file=op)
    print ('    VW_P2 ${INV_VW_P2};', file=op)

#***************************************************************************************************
#***************************************************************************************************

def write_tariff (op):
    """Writes tariff information to billing meters

    Args:
        op (file): an open GridLAB-D input file
    """
    print ('  bill_mode', ConfigDict['billing']['bill_mode']['value'] + ';', file=op)
    print ('  price', '{:.4f}'.format (ConfigDict['billing']['kwh_price']['value']) + ';', file=op)
    print ('  monthly_fee', '{:.2f}'.format (ConfigDict['billing']['monthly_fee']['value']) + ';', file=op)
    print ('  bill_day 1;', file=op)
    if 'TIERED' in ConfigDict['billing']['bill_mode']['value']:
        if ConfigDict['billing']['tier1_energy']['value'] > 0.0:
            print ('  first_tier_energy', '{:.1f}'.format (ConfigDict['billing']['tier1_energy']['value']) + ';', file=op)
            print ('  first_tier_price', '{:.6f}'.format (ConfigDict['billing']['tier1_price']['value']) + ';', file=op)
        if ConfigDict['billing']['tier2_energy']['value'] > 0.0:
            print ('  second_tier_energy', '{:.1f}'.format (ConfigDict['billing']['tier2_energy']['value']) + ';', file=op)
            print ('  second_tier_price', '{:.6f}'.format (ConfigDict['billing']['tier2_price']['value']) + ';', file=op)
        if ConfigDict['billing']['tier3_energy']['value'] > 0.0:
            print ('  third_tier_energy', '{:.1f}'.format (ConfigDict['billing']['tier3_energy']['value']) + ';', file=op)
            print ('  third_tier_price', '{:.6f}'.format (ConfigDict['billing']['tier3_price']['value']) + ';', file=op)

#***************************************************************************************************
#***************************************************************************************************

def obj(parent,model,line,itr,oidh,octr):
    """Store an object in the model structure

    Args:
        parent (str): name of parent object (used for nested object defs)
        model (dict): dictionary model structure
        line (str): glm line containing the object definition
        itr (iter): iterator over the list of lines
        oidh (dict): hash of object id's to object names
        octr (int): object counter

    Returns:
        str, int: the current line and updated octr
    """
    octr += 1
    # Identify the object type
    m = re.search('object ([^:{\s]+)[:{\s]',line,re.IGNORECASE)
    type = m.group(1)
    # If the object has an id number, store it
    n = re.search('object ([^:]+:[^{\s]+)',line,re.IGNORECASE)
    if n:
        oid = n.group(1)
    line = next(itr)
    # Collect parameters
    oend = 0
    oname = None
    params = {}
    if parent is not None:
        params['parent'] = parent
    while not oend:
        m = re.match('\s*(\S+) ([^;{]+)[;{]',line)
        if m:
            # found a parameter
            param = m.group(1)
            val = m.group(2)
            intobj = 0
            if param == 'name':
                oname = ConfigDict['name_prefix']['value'] + val
            elif param == 'object':
                # found a nested object
                intobj += 1
                if oname is None:
                    print('ERROR: nested object defined before parent name')
                    quit()
                line,octr = obj(oname,model,line,itr,oidh,octr)
            elif re.match('object',val):
                # found an inline object
                intobj += 1
                line,octr = obj(None,model,line,itr,oidh,octr)
                params[param] = 'ID_'+str(octr)
            else:
                params[param] = val
        if re.search('}',line):
            if intobj:
                intobj -= 1
                line = next(itr)
            else:
                oend = 1
        else:
            line = next(itr)
    # If undefined, use a default name
    if oname is None:
        oname = ConfigDict['name_prefix']['value'] + 'ID_'+str(octr)
    oidh[oname] = oname
    # Hash an object identifier to the object name
    if n:
        oidh[oid] = oname
    # Add the object to the model
    if type not in model:
        # New object type
        model[type] = {}
    model[type][oname] = {}
    for param in params:
        model[type][oname][param] = params[param]
    return line,octr

#***************************************************************************************************
#***************************************************************************************************
def write_link_class (model, h, t, seg_loads, op, want_metrics=False):
  """Write a GridLAB-D link (i.e. edge) class

  Args:
      model (dict): the parsed GridLAB-D model
      h (dict): the object ID hash
      t (str): the GridLAB-D class
      seg_loads (dict) : a dictionary of downstream loads for each link
      op (file): an open GridLAB-D input file
  """
  if t in model:
    for o in model[t]:
#            print('object ' + t + ':' + o + ' {', file=op)
      print('object ' + t + ' {', file=op)
      print('  name ' + o + ';', file=op)
      if o in seg_loads:
        print('// downstream', '{:.2f}'.format(seg_loads[o][0]), 'kva on', seg_loads[o][1], file=op)
      for p in model[t][o]:
        if ':' in model[t][o][p]:
          print ('  ' + p + ' ' + h[model[t][o][p]] + ';', file=op)
        else:
          print ('  ' + p + ' ' + model[t][o][p] + ';', file=op)
      #if want_metrics and metrics_interval > 0:
      if want_metrics and ConfigDict['metrics_interval']['value'] > 0:
        print ('  object metrics_collector {', file=op)
        #print ('    interval', str(metrics_interval) + ';', file=op)
        print ('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
        print ('  };', file=op)
      print('}', file=op)

#***************************************************************************************************
#***************************************************************************************************

def write_local_triplex_configurations (op):
  """Write a 4/0 AA triplex configuration

  Args:
    op (file): an open GridLAB-D input file
  """
  for row in ConfigDict['triplex_conductors']:
    print ('object triplex_line_conductor {', file=op)
    print ('  name', ConfigDict['name_prefix']['value'] + row + ';', file=op)
    print ('  resistance', str(ConfigDict['triplex_conductors'][row]['resistance']) + ';', file=op)
    print ('  geometric_mean_radius', str(ConfigDict['triplex_conductors'][row]['geometric_mean_radius']) + ';', file=op)
    rating_str = str(ConfigDict['triplex_conductors'][row]['rating']) 
    print ('  rating.summer.continuous', rating_str + ';', file=op)
    print ('  rating.summer.emergency', rating_str + ';', file=op)
    print ('  rating.winter.continuous', rating_str + ';', file=op)
    print ('  rating.winter.emergency', rating_str + ';', file=op)
    print ('}', file=op)
  for row in ConfigDict['triplex_configurations']:
    print ('object triplex_line_configuration {', file=op)
    print ('  name ', ConfigDict['name_prefix']['value'] + row + ';', file=op)
    print ('  conductor_1 ', ConfigDict['name_prefix']['value'] + ConfigDict['triplex_configurations'][row]['conductor_1'] + ';', file=op)
    print ('  conductor_2 ', ConfigDict['name_prefix']['value'] + ConfigDict['triplex_configurations'][row]['conductor_2']  + ';', file=op)
    print ('  conductor_N ', ConfigDict['name_prefix']['value'] + ConfigDict['triplex_configurations'][row]['conductor_N']  + ';', file=op) # Need to validate this as the correct value. Just putting this in as a placeholder for now.
    print ('  insulation_thickness ', str(ConfigDict['triplex_configurations'][row]['insulation']) + ';', file=op)
    print ('  diameter ', str(ConfigDict['triplex_configurations'][row]['diameter']) + ';', file=op)
    print ('}', file=op)

#***************************************************************************************************
#***************************************************************************************************

def connect_ercot_houses (model, h, op, vln, vsec):
    """For the reduced-order ERCOT feeders, add houses and a large service transformer to the load points

    Args:
        model (dict): the parsed GridLAB-D model
        h (dict): the object ID hash
        op (file): an open GridLAB-D input file
        vln (float): the primary line-to-neutral voltage
        vsec (float): the secondary line-to-neutral voltage
    """
    for key in ConfigDict['house_nodes']['value']:
#        bus = key[:-2]
        bus = ConfigDict['house_nodes']['value'][key][6]
        phs = ConfigDict['house_nodes']['value'][key][3]
        nh = ConfigDict['house_nodes']['value'][key][0]
        xfkva = Find1PhaseXfmrKva (6.0 * nh)
        if xfkva > 100.0:
            npar = int (xfkva / 100.0 + 0.5)
            xfkva = 100.0
        elif xfkva <= 0.0:
            xfkva = 100.0
            npar = int (0.06 * nh + 0.5)
        else:
            npar = 1
#        print (key, bus, phs, nh, xfkva, npar)
        # write the service transformer==>TN==>TPX==>TM for all houses
        kvat = npar * xfkva
        row = Find1PhaseXfmr (xfkva)
        print ('object transformer_configuration {', file=op)
        print ('  name ' + key + '_xfconfig;', file=op)
        print ('  power_rating ' + format(kvat, '.2f') + ';', file=op)
        if 'A' in phs:
            print ('  powerA_rating ' + format(kvat, '.2f') + ';', file=op)
        elif 'B' in phs:
            print ('  powerB_rating ' + format(kvat, '.2f') + ';', file=op)
        elif 'C' in phs:
            print ('  powerC_rating ' + format(kvat, '.2f') + ';', file=op)
        print ('  install_type PADMOUNT;', file=op)
        print ('  connect_type SINGLE_PHASE_CENTER_TAPPED;', file=op)
        print ('  primary_voltage ' + str(vln) + ';', file=op)
        print ('  secondary_voltage ' + format(vsec, '.1f') + ';', file=op)
        print ('  resistance ' + format(row[1] * 0.5, '.5f') + ';', file=op)
        print ('  resistance1 ' + format(row[1], '.5f') + ';', file=op)
        print ('  resistance2 ' + format(row[1], '.5f') + ';', file=op)
        print ('  reactance ' + format(row[2] * 0.8, '.5f') + ';', file=op)
        print ('  reactance1 ' + format(row[2] * 0.4, '.5f') + ';', file=op)
        print ('  reactance2 ' + format(row[2] * 0.4, '.5f') + ';', file=op)
        print ('  shunt_resistance ' + format(1.0 / row[3], '.2f') + ';', file=op)
        print ('  shunt_reactance ' + format(1.0 / row[4], '.2f') + ';', file=op)
        print ('}', file=op)
        print ('object transformer {', file=op)
        print ('  name ' + key + '_xf;', file=op)
        print ('  phases ' + phs + 'S;', file=op)
        print ('  from ' + bus + ';', file=op)
        print ('  to ' + key + '_tn;', file=op)
        print ('  configuration ' + key + '_xfconfig;', file=op)
        print ('}', file=op)
        print ('object triplex_line_configuration {', file=op)
        print ('  name ' + key + '_tpxconfig;', file=op)
        zs = format (ConfigDict['tpxR11']['value']/nh, '.5f') + '+' + format (ConfigDict['tpxX11']['value']/nh, '.5f') + 'j;'
        zm = format (ConfigDict['tpxR12']['value']/nh, '.5f') + '+' + format (ConfigDict['tpxX12']['value']/nh, '.5f') + 'j;'
        amps = format (ConfigDict['tpxAMP']['value'] * nh, '.1f') + ';'
        print ('  z11 ' + zs, file=op)
        print ('  z22 ' + zs, file=op)
        print ('  z12 ' + zm, file=op)
        print ('  z21 ' + zm, file=op)
        print ('  rating.summer.continuous ' + amps, file=op)
        print ('}', file=op)
        print ('object triplex_line {', file=op)
        print ('  name ' + key + '_tpx;', file=op)
        print ('  phases ' + phs + 'S;', file=op)
        print ('  from ' + key + '_tn;', file=op)
        print ('  to ' + key + '_mtr;', file=op)
        print ('  length 50;', file=op)
        print ('  configuration ' + key + '_tpxconfig;', file=op)
        print ('}', file=op)
        if 'A' in phs:
            vstart = str(vsec) + '+0.0j;'
        elif 'B' in phs:
            vstart = format(-0.5*vsec,'.2f') + format(-0.866025*vsec,'.2f') + 'j;'
        else:
            vstart = format(-0.5*vsec,'.2f') + '+' + format(0.866025*vsec,'.2f') + 'j;'
        print ('object triplex_node {', file=op)
        print ('  name ' + key + '_tn;', file=op)
        print ('  phases ' + phs + 'S;', file=op)
        print ('  voltage_1 ' + vstart, file=op)
        print ('  voltage_2 ' + vstart, file=op)
        print ('  voltage_N 0;', file=op)
        print ('  nominal_voltage ' + format(vsec, '.1f') + ';', file=op)
        print ('}', file=op)
        print ('object triplex_meter {', file=op)
        print ('  name ' + key + '_mtr;', file=op)
        print ('  phases ' + phs + 'S;', file=op)
        print ('  voltage_1 ' + vstart, file=op)
        print ('  voltage_2 ' + vstart, file=op)
        print ('  voltage_N 0;', file=op)
        print ('  nominal_voltage ' + format(vsec, '.1f') + ';', file=op)
        write_tariff (op)
        if ConfigDict['metrics_interval']['value'] > 0:
            print ('  object metrics_collector {', file=op)
            print ('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
            print ('  };', file=op)
        print ('}', file=op)

#***************************************************************************************************
#***************************************************************************************************

def connect_ercot_commercial(op):
  """For the reduced-order ERCOT feeders, add a billing meter to the commercial load points, except small ZIPLOADs

  Args:
      op (file): an open GridLAB-D input file
  """
  meters_added = set()
  for key in ConfigDict['comm_loads']['value']:
    mtr = ConfigDict['comm_loads']['value'][key][0]
    comm_type = ConfigDict['comm_loads']['value'][key][1]
    if comm_type == 'ZIPLOAD':
      continue
    phases = ConfigDict['comm_loads']['value'][key][5]
    vln = float(ConfigDict['comm_loads']['value'][key][6])
    idx = mtr.rfind('_')
    parent = mtr[:idx]

    if mtr not in meters_added:
      meters_added.add(mtr)
      print ('object meter {', file=op)
      print ('  name ' + mtr + ';', file=op)
      print ('  parent ' + parent + ';', file=op)
      print ('  phases ' + phases + ';', file=op)
      print ('  nominal_voltage ' + format(vln, '.1f') + ';', file=op)
      write_tariff (op)
      if ConfigDict['metrics_interval']['value'] > 0:
          print ('  object metrics_collector {', file=op)
          print ('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
          print ('  };', file=op)
      print ('}', file=op)

#***************************************************************************************************
#***************************************************************************************************

def write_ercot_small_loads(basenode, op, vnom):
  """For the reduced-order ERCOT feeders, write loads that are too small for houses

  Args:
    basenode (str): the GridLAB-D node name
    op (file): an open GridLAB-D input file
    vnom (float): the primary line-to-neutral voltage
  """
  kva = float(ConfigDict['small_nodes']['value'][basenode][0])
  phs = ConfigDict['small_nodes']['value'][basenode][1]
  parent = ConfigDict['small_nodes']['value'][basenode][2]
  cls = ConfigDict['small_nodes']['value'][basenode][3]

  if 'A' in phs:
      vstart = '  voltage_A ' + str(vnom) + '+0.0j;'
      constpower = '  constant_power_A_real ' + format (1000.0 * kva, '.2f') + ';'
  elif 'B' in phs:
      vstart = '  voltage_B ' + format(-0.5*vnom,'.2f') + format(-0.866025*vnom,'.2f') + 'j;'
      constpower = '  constant_power_B_real ' + format (1000.0 * kva, '.2f') + ';'
  else:
      vstart = '  voltage_C ' + format(-0.5*vnom,'.2f') + '+' + format(0.866025*vnom,'.2f') + 'j;'
      constpower = '  constant_power_C_real ' + format (1000.0 * kva, '.2f') + ';'

  print ('object load {', file=op)
  print ('  name', basenode + ';', file=op)
  print ('  parent', parent + ';', file=op)
  print ('  phases', phs + ';', file=op)
  print ('  nominal_voltage ' + str(vnom) + ';', file=op)
  print ('  load_class ' + cls + ';', file=op)
  print (vstart, file=op)
  print ('  //', '{:.3f}'.format(kva), 'kva is less than 1/2 avg_house', file=op)
  print (constpower, file=op)
  print ('}', file=op)

#***************************************************************************************************
#***************************************************************************************************
# look at primary loads, not the service transformers
def identify_ercot_houses (model, h, t, avgHouse, rgn):
    """For the reduced-order ERCOT feeders, scan each primary load to determine the number of houses it should have

    Args:
        model (dict): the parsed GridLAB-D model
        h (dict): the object ID hash
        t (str): the GridLAB-D class name to scan
        avgHouse (float): the average house load in kva
        rgn (int): the region number, 1..5
    """
    print ('Average ERCOT House', avgHouse, rgn)
    total_houses = {'A': 0, 'B': 0, 'C': 0}
    total_small =  {'A': 0, 'B': 0, 'C': 0}
    total_small_kva =  {'A': 0, 'B': 0, 'C': 0}
    total_sf = 0
    total_apt = 0
    total_mh = 0
    if t in model:
        for o in model[t]:
            name = o
            node = o
            parent = model[t][o]['parent']
            for phs in ['A', 'B', 'C']:
                tok = 'constant_power_' + phs
                key = node + '_' + phs
                if tok in model[t][o]:
                    kva = parse_kva (model[t][o][tok])
                    nh = 0
                    cls = 'U'
                    # don't populate houses onto A, C, I or U load_class nodes
                    if 'load_class' in model[t][o]:
                        cls = model[t][o]['load_class']
                        if cls == 'R':
                            if (kva > 1.0):
                                nh = int ((kva / avgHouse) + 0.5)
                                total_houses[phs] += nh
                    if nh > 0:
                        lg_v_sm = kva / avgHouse - nh # >0 if we rounded down the number of houses
                        bldg, ti = selectResidentialBuilding (ConfigDict['rgnThermalPct']['value'][rgn-1], np.random.uniform (0, 1))
                        if bldg == 0:
                            total_sf += nh
                        elif bldg == 1:
                            total_apt += nh
                        else:
                            total_mh += nh
                        ConfigDict['house_nodes']['value'][key] = [nh, rgn, lg_v_sm, phs, bldg, ti, parent] # parent is the primary node, only for ERCOT
                    elif kva > 0.1:
                        total_small[phs] += 1
                        total_small_kva[phs] += kva
                        ConfigDict['small_nodes']['value'][key] = [kva, phs, parent, cls] # parent is the primary node, only for ERCOT
    for phs in ['A', 'B', 'C']:
        print ('phase', phs, ':', total_houses[phs], 'Houses and', total_small[phs],
               'Small Loads totaling', '{:.2f}'.format (total_small_kva[phs]), 'kva')
    print (len(ConfigDict['house_nodes']['value']), 'primary house nodes, [SF,APT,MH]=', total_sf, total_apt, total_mh)
    for i in range(6):
        ConfigDict['heating_bins']['value'][0][i] = round (total_sf * ConfigDict['bldgHeatingSetpoints']['value'][0][i][0] + 0.5)
        ConfigDict['heating_bins']['value'][1][i] = round (total_apt * ConfigDict['bldgHeatingSetpoints']['value'][1][i][0] + 0.5)
        ConfigDict['heating_bins']['value'][2][i] = round (total_mh * ConfigDict['bldgHeatingSetpoints']['value'][2][i][0] + 0.5)
        ConfigDict['cooling_bins']['value'][0][i] = round (total_sf * ConfigDict['bldgCoolingSetpoints']['value'][0][i][0] + 0.5)
        ConfigDict['cooling_bins']['value'][1][i] = round (total_apt * ConfigDict['bldgCoolingSetpoints']['value'][1][i][0] + 0.5)
        ConfigDict['cooling_bins']['value'][2][i] = round (total_mh * ConfigDict['bldgCoolingSetpoints']['value'][2][i][0] + 0.5)
    print ('cooling bins target', ConfigDict['cooling_bins']['value'])
    print ('heating bins target', ConfigDict['heating_bins']['value'])

#***************************************************************************************************
#***************************************************************************************************

def replace_commercial_loads (model, h, t, avgBuilding):
  """For the full-order feeders, scan each load with load_class==C to determine the number of zones it should have

  Args:
      model (dict): the parsed GridLAB-D model
      h (dict): the object ID hash
      t (str): the GridLAB-D class name to scan
      avgBuilding (float): the average building in kva
  """
  print ('Average Commercial Building', avgBuilding)
  total_commercial = 0
  total_comm_kva = 0
  total_comm_zones = 0
  total_zipload = 0
  total_office = 0
  total_bigbox = 0
  total_stripmall = 0
  if t in model:
    for o in list(model[t].keys()):
      if 'load_class' in model[t][o]:
        if model[t][o]['load_class'] == 'C':
          kva = accumulate_load_kva (model[t][o])
          total_commercial += 1
          total_comm_kva += kva
          vln = float(model[t][o]['nominal_voltage'])
          nphs = 0
          phases = model[t][o]['phases']
          if 'A' in phases:
            nphs += 1
          if 'B' in phases:
            nphs += 1
          if 'C' in phases:
            nphs += 1
          nzones = int ((kva / avgBuilding) + 0.5)
          total_comm_zones += nzones
          if nzones > 14 and nphs == 3:
            comm_type = 'OFFICE'
            total_office += 1
          elif nzones > 5 and nphs > 1:
            comm_type = 'BIGBOX'
            total_bigbox += 1
          elif nzones > 0:
            comm_type = 'STRIPMALL'
            total_stripmall += 1
          else:
            comm_type = 'ZIPLOAD'
            total_zipload += 1
          mtr = model[t][o]['parent']
          if ConfigDict['forERCOT']['value'] == "True":
          # the parent node is actually a meter, but we have to add the tariff and metrics_collector unless only ZIPLOAD
            mtr = model[t][o]['parent'] # + '_mtr'
            if comm_type != 'ZIPLOAD':
              extra_billing_meters.add(mtr)
          else:
            extra_billing_meters.add(mtr)
          ConfigDict['comm_loads']['value'][o] = [mtr, comm_type, nzones, kva, nphs, phases, vln, total_commercial]
          model[t][o]['groupid'] = comm_type + '_' + str(nzones)
          del model[t][o]
  print ('found', total_commercial, 'commercial loads totaling ', '{:.2f}'.format(total_comm_kva), 'KVA')
  print (total_office, 'offices,', total_bigbox, 'bigbox retail,', total_stripmall, 'strip malls,',
         total_zipload, 'ZIP loads')
  print (total_comm_zones, 'total commercial HVAC zones')

#***************************************************************************************************
#***************************************************************************************************

def identify_xfmr_houses (model, h, t, seg_loads, avgHouse, rgn):
    """For the full-order feeders, scan each service transformer to determine the number of houses it should have

    Args:
        model (dict): the parsed GridLAB-D model
        h (dict): the object ID hash
        t (str): the GridLAB-D class name to scan
        seg_loads (dict): dictionary of downstream load (kva) served by each GridLAB-D link
        avgHouse (float): the average house load in kva
        rgn (int): the region number, 1..5
    """
    print ('Average House', avgHouse)
    total_houses = 0
    total_sf = 0
    total_apt = 0
    total_mh = 0
    total_small = 0
    total_small_kva = 0
    if t in model:
        for o in model[t]:
            if o in seg_loads:
                tkva = seg_loads[o][0]
                phs = seg_loads[o][1]
                if 'S' in phs:
                    nhouse = int ((tkva / avgHouse) + 0.5) # round to nearest int
                    name = o
                    node = model[t][o]['to']
                    if nhouse <= 0:
                        total_small += 1
                        total_small_kva += tkva
                        ConfigDict['small_nodes'][node] = [tkva,phs]
                    else:
                        total_houses += nhouse
                        lg_v_sm = tkva / avgHouse - nhouse # >0 if we rounded down the number of houses
                        bldg, ti = selectResidentialBuilding (ConfigDict['rgnThermalPct']['value'][rgn-1], np.random.uniform (0, 1))
                        if bldg == 0:
                            total_sf += nhouse
                        elif bldg == 1:
                            total_apt += nhouse
                        else:
                            total_mh += nhouse
                        ConfigDict['house_nodes']['value'][node] = [nhouse, rgn, lg_v_sm, phs, bldg, ti]
    print (total_small, 'small loads totaling', '{:.2f}'.format (total_small_kva), 'kva')
    print (total_houses, 'houses on', len(ConfigDict['house_nodes']['value']), 'transformers, [SF,APT,MH]=', total_sf, total_apt, total_mh)
    for i in range(6):
        ConfigDict['heating_bins']['value'][0][i] = round (total_sf * ConfigDict['bldgHeatingSetpoints']['value'][0][i][0] + 0.5)
        ConfigDict['heating_bins']['value'][1][i] = round (total_apt * ConfigDict['bldgHeatingSetpoints']['value'][1][i][0] + 0.5)
        ConfigDict['heating_bins']['value'][2][i] = round (total_mh * ConfigDict['bldgHeatingSetpoints']['value'][2][i][0] + 0.5)
        ConfigDict['cooling_bins']['value'][0][i] = round (total_sf * ConfigDict['bldgCoolingSetpoints']['value'][0][i][0] + 0.5)
        ConfigDict['cooling_bins']['value'][1][i] = round (total_apt * ConfigDict['bldgCoolingSetpoints']['value'][1][i][0] + 0.5)
        ConfigDict['cooling_bins']['value'][2][i] = round (total_mh * ConfigDict['bldgCoolingSetpoints']['value'][2][i][0] + 0.5)
    print ('cooling bins target', ConfigDict['cooling_bins']['value'])
    print ('heating bins target', ConfigDict['heating_bins']['value'])

#***************************************************************************************************
#***************************************************************************************************

def write_small_loads(basenode, op, vnom):
  """Write loads that are too small for a house, onto a node

  Args:
    basenode (str): GridLAB-D node name
    op (file): open file to write to
    vnom (float): nominal line-to-neutral voltage at basenode
  """
  kva = float(ConfigDict['small_nodes']['value'][basenode][0])
  phs = ConfigDict['small_nodes']['value'][basenode][1]

  if 'A' in phs:
      vstart = str(vnom) + '+0.0j'
  elif 'B' in phs:
      vstart = format(-0.5*vnom,'.2f') + format(-0.866025*vnom,'.2f') + 'j'
  else:
      vstart = format(-0.5*vnom,'.2f') + '+' + format(0.866025*vnom,'.2f') + 'j'

  tpxname = basenode + '_tpx_1'
  mtrname = basenode + '_mtr_1'
  loadname = basenode + '_load_1'
  print ('object triplex_node {', file=op)
  print ('  name', basenode + ';', file=op)
  print ('  phases', phs + ';', file=op)
  print ('  nominal_voltage ' + str(vnom) + ';', file=op)
  print ('  voltage_1 ' + vstart + ';', file=op)
  print ('  voltage_2 ' + vstart + ';', file=op)
  print ('}', file=op)
  print ('object triplex_line {', file=op)
  print ('  name', tpxname + ';', file=op)
  print ('  from', basenode + ';', file=op)
  print ('  to', mtrname + ';', file=op)
  print ('  phases', phs + ';', file=op)
  print ('  length 30;', file=op)
  print ('  configuration', ConfigDict['triplex_configurations'][0][0] + ';', file=op)
  print ('}', file=op)
  print ('object triplex_meter {', file=op)
  print ('  name', mtrname + ';', file=op)
  print ('  phases', phs + ';', file=op)
  print ('  meter_power_consumption 1+7j;', file=op)
  write_tariff (op)
  print ('  nominal_voltage ' + str(vnom) + ';', file=op)
  print ('  voltage_1 ' + vstart + ';', file=op)
  print ('  voltage_2 ' + vstart + ';', file=op)
  if ConfigDict['metrics_interval']['value'] > 0:
    print ('  object metrics_collector {', file=op)
    print ('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
    print ('  };', file=op)
  print ('}', file=op)
  print ('object triplex_load {', file=op)
  print ('  name', loadname + ';', file=op)
  print ('  parent', mtrname + ';', file=op)
  print ('  phases', phs + ';', file=op)
  print ('  nominal_voltage ' + str(vnom) + ';', file=op)
  print ('  voltage_1 ' + vstart + ';', file=op)
  print ('  voltage_2 ' + vstart + ';', file=op)
  print ('  //', '{:.3f}'.format(kva), 'kva is less than 1/2 avg_house', file=op)
  print ('  power_12_real 10.0;', file=op)
  print ('  power_12_reac 8.0;', file=op)
  print ('}', file=op)

#***************************************************************************************************
#***************************************************************************************************

def write_one_commercial_zone(bldg, op):
  """Write one pre-configured commercial zone as a house

  Args:
      bldg: dictionary of GridLAB-D house and zipload attributes
      op (file): open file to write to
  """
  print ('object house {', file=op)
  print ('  name', bldg['zonename'] + ';', file=op)
  print ('  parent', bldg['parent'] + ';', file=op)
  print ('  groupid', bldg['groupid'] + ';', file=op)
  print ('  motor_model BASIC;', file=op)
  print ('  schedule_skew {:.0f};'.format(bldg['skew_value']), file=op)
  print ('  floor_area {:.0f};'.format(bldg['floor_area']), file=op)
  print ('  design_internal_gains {:.0f};'.format(bldg['int_gains'] * bldg['floor_area'] * 3.413), file=op)
  print ('  number_of_doors {:.0f};'.format(bldg['no_of_doors']), file=op)
  print ('  aspect_ratio {:.2f};'.format(bldg['aspect_ratio']), file=op)
  print ('  total_thermal_mass_per_floor_area {:1.2f};'.format(bldg['thermal_mass_per_floor_area']), file=op)
  print ('  interior_surface_heat_transfer_coeff {:1.2f};'.format(bldg['surface_heat_trans_coeff']), file=op)
  print ('  interior_exterior_wall_ratio {:.2f};'.format(bldg['interior_exterior_wall_ratio']), file=op)
  print ('  exterior_floor_fraction {:.3f};'.format(bldg['exterior_floor_fraction']), file=op)
  print ('  exterior_ceiling_fraction {:.3f};'.format(bldg['exterior_ceiling_fraction']), file=op)
  print ('  Rwall {:2.1f};'.format(bldg['Rwall']), file=op)
  print ('  Rroof {:2.1f};'.format(bldg['Rroof']), file=op)
  print ('  Rfloor {:.2f};'.format(bldg['Rfloor']), file=op)
  print ('  Rdoors {:2.1f};'.format(bldg['Rdoors']), file=op)
  print ('  exterior_wall_fraction {:.2f};'.format(bldg['exterior_wall_fraction']), file=op)
  print ('  glazing_layers {:s};'.format(bldg['glazing_layers']), file=op)
  print ('  glass_type {:s};'.format(bldg['glass_type']), file=op)
  print ('  glazing_treatment {:s};'.format(bldg['glazing_treatment']), file=op)
  print ('  window_frame {:s};'.format(bldg['window_frame']), file=op)
  print ('  airchange_per_hour {:.2f};'.format(bldg['airchange_per_hour']), file=op)
  print ('  window_wall_ratio {:0.3f};'.format(bldg['window_wall_ratio']), file=op)
  print ('  heating_system_type {:s};'.format(bldg['heat_type']), file=op)
  print ('  auxiliary_system_type {:s};'.format(bldg['aux_type']), file=op)
  print ('  fan_type {:s};'.format(bldg['fan_type']), file=op)
  print ('  cooling_system_type {:s};'.format(bldg['cool_type']), file=op)
  print ('  air_temperature {:.2f};'.format(bldg['init_temp']), file=op)
  print ('  mass_temperature {:.2f};'.format(bldg['init_temp']), file=op)
  print ('  over_sizing_factor {:.1f};'.format(bldg['os_rand']), file=op)
  print ('  cooling_COP {:2.2f};'.format(bldg['COP_A']), file=op)
  print ('  cooling_setpoint 80.0; // {:s}_cooling'.format(bldg['base_schedule']), file=op)
  print ('  heating_setpoint 60.0; // {:s}_heating'.format(bldg['base_schedule']), file=op)
  print ('  object ZIPload { // lights', file=op)
  print ('    schedule_skew {:.0f};'.format(bldg['skew_value']), file=op)
  print ('    heatgain_fraction 1.0;', file=op)
  print ('    power_fraction {:.2f};'.format(bldg['c_p_frac']), file=op)
  print ('    impedance_fraction {:.2f};'.format(bldg['c_z_frac']), file=op)
  print ('    current_fraction {:.2f};'.format(bldg['c_i_frac']), file=op)
  print ('    power_pf {:.2f};'.format(bldg['c_p_pf']), file=op)
  print ('    current_pf {:.2f};'.format(bldg['c_i_pf']), file=op)
  print ('    impedance_pf {:.2f};'.format(bldg['c_z_pf']), file=op)
  print ('    base_power {:s}_lights*{:.2f};'.format(bldg['base_schedule'], bldg['adj_lights']), file=op)
  print ('  };', file=op)
  print ('  object ZIPload { // plug loads', file=op)
  print ('    schedule_skew {:.0f};'.format(bldg['skew_value']), file=op)
  print ('    heatgain_fraction 1.0;', file=op)
  print ('    power_fraction {:.2f};'.format(bldg['c_p_frac']), file=op)
  print ('    impedance_fraction {:.2f};'.format(bldg['c_z_frac']), file=op)
  print ('    current_fraction {:.2f};'.format(bldg['c_i_frac']), file=op)
  print ('    power_pf {:.2f};'.format(bldg['c_p_pf']), file=op)
  print ('    current_pf {:.2f};'.format(bldg['c_i_pf']), file=op)
  print ('    impedance_pf {:.2f};'.format(bldg['c_z_pf']), file=op)
  print ('    base_power {:s}_plugs*{:.2f};'.format(bldg['base_schedule'], bldg['adj_plugs']), file=op)
  print ('  };', file=op)
  print ('  object ZIPload { // gas waterheater', file=op)
  print ('    schedule_skew {:.0f};'.format(bldg['skew_value']), file=op)
  print ('    heatgain_fraction 1.0;', file=op)
  print ('    power_fraction 0;', file=op)
  print ('    impedance_fraction 0;', file=op)
  print ('    current_fraction 0;', file=op)
  print ('    power_pf 1;', file=op)
  print ('    base_power {:s}_gas*{:.2f};'.format(bldg['base_schedule'], bldg['adj_gas']), file=op)
  print ('  };', file=op)
  print ('  object ZIPload { // exterior lights', file=op)
  print ('    schedule_skew {:.0f};'.format(bldg['skew_value']), file=op)
  print ('    heatgain_fraction 0.0;', file=op)
  print ('    power_fraction {:.2f};'.format(bldg['c_p_frac']), file=op)
  print ('    impedance_fraction {:.2f};'.format(bldg['c_z_frac']), file=op)
  print ('    current_fraction {:.2f};'.format(bldg['c_i_frac']), file=op)
  print ('    power_pf {:.2f};'.format(bldg['c_p_pf']), file=op)
  print ('    current_pf {:.2f};'.format(bldg['c_i_pf']), file=op)
  print ('    impedance_pf {:.2f};'.format(bldg['c_z_pf']), file=op)
  print ('    base_power {:s}_exterior*{:.2f};'.format(bldg['base_schedule'], bldg['adj_ext']), file=op)
  print ('  };', file=op)
  print ('  object ZIPload { // occupancy', file=op)
  print ('    schedule_skew {:.0f};'.format(bldg['skew_value']), file=op)
  print ('    heatgain_fraction 1.0;', file=op)
  print ('    power_fraction 0;', file=op)
  print ('    impedance_fraction 0;', file=op)
  print ('    current_fraction 0;', file=op)
  print ('    power_pf 1;', file=op)
  print ('    base_power {:s}_occupancy*{:.2f};'.format(bldg['base_schedule'], bldg['adj_occ']), file=op)
  print ('  };', file=op)
  if ConfigDict['metrics_interval']['value'] > 0:
    print ('  object metrics_collector {', file=op)
    print ('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
    print ('  };', file=op)
  print ('}', file=op)

#***************************************************************************************************
#***************************************************************************************************

def write_commercial_loads(rgn, key, op):
  """Put commercial building zones and ZIP loads into the model

  Args:
      rgn (int): region 1..5 where the building is located
      key (str): GridLAB-D load name that is being replaced
      op (file): open file to write to
  """
  mtr = ConfigDict['comm_loads']['value'][key][0]
  comm_type = ConfigDict['comm_loads']['value'][key][1]
  nz = int(ConfigDict['comm_loads']['value'][key][2])
  kva = float(ConfigDict['comm_loads']['value'][key][3])
  nphs = int(ConfigDict['comm_loads']['value'][key][4])
  phases = ConfigDict['comm_loads']['value'][key][5]
  vln = float(ConfigDict['comm_loads']['value'][key][6])
  loadnum = int(ConfigDict['comm_loads']['value'][key][7])

  bldg = {}
  bldg['parent'] = key
  bldg['mtr'] = mtr
  bldg['groupid'] = comm_type + '_' + str(loadnum)

  print ('// load', key, 'mtr', bldg['mtr'], 'type', comm_type, 'nz', nz, 'kva', '{:.3f}'.format(kva),
         'nphs', nphs, 'phases', phases, 'vln', '{:.3f}'.format(vln), file=op)

  bldg['fan_type'] = 'ONE_SPEED'
  bldg['heat_type'] = 'GAS'
  bldg['cool_type'] = 'ELECTRIC'
  bldg['aux_type'] = 'NONE'
  bldg['no_of_stories'] = 1
  bldg['surface_heat_trans_coeff'] = 0.59
  bldg['oversize'] = ConfigDict['over_sizing_factor']['value'][rgn-1]
  bldg['glazing_layers'] = 'TWO'
  bldg['glass_type'] = 'GLASS'
  bldg['glazing_treatment'] = 'LOW_S'
  bldg['window_frame'] = 'NONE'
  bldg['c_z_frac'] = ConfigDict['c_z_frac']['value']
  bldg['c_i_frac'] = ConfigDict['c_i_frac']['value']
  bldg['c_p_frac'] = c_p_frac
  bldg['c_z_pf'] = ConfigDict['c_z_pf']['value']
  bldg['c_i_pf'] = ConfigDict['c_i_pf']['value']
  bldg['c_p_pf'] = ConfigDict['c_p_pf']['value']

  if comm_type == 'OFFICE':
    bldg['ceiling_height'] = 13.
    bldg['airchange_per_hour'] = 0.69
    bldg['Rroof'] = 19.
    bldg['Rwall'] = 18.3
    bldg['Rfloor'] = 46.
    bldg['Rdoors'] = 3.
    bldg['int_gains'] = 3.24  # W/sf
    bldg['thermal_mass_per_floor_area'] = 1 # TODO
    bldg['exterior_ceiling_fraction'] = 1 # TODO
    bldg['base_schedule'] = 'office'
    num_offices = int(round(nz/15))  # each with 3 floors of 5 zones
    for jjj in range(num_offices):
      floor_area_choose = 40000. * (0.5 * np.random.random() + 0.5)
      for floor in range(1, 4):
        bldg['skew_value'] = randomize_commercial_skew()
        total_depth = math.sqrt(floor_area_choose / (3. * 1.5))
        total_width = 1.5 * total_depth
        if floor == 3:
          bldg['exterior_ceiling_fraction'] = 1
        else:
          bldg['exterior_ceiling_fraction'] = 0
        for zone in range(1, 6):
          if zone == 5:
            bldg['window_wall_ratio'] = 0  # this was not in the CCSI version
            bldg['exterior_wall_fraction'] = 0
            w = total_depth - 30.
            d = total_width - 30.
          else:
            bldg['window_wall_ratio'] = 0.33
            d = 15.
            if zone == 1 or zone == 3:
              w = total_width - 15.
            else:
              w = total_depth - 15.
            bldg['exterior_wall_fraction'] = w / (2. * (w + d))

          floor_area = w * d
          bldg['floor_area'] = floor_area
          bldg['aspect_ratio'] = w / d

          if floor > 1:
            bldg['exterior_floor_fraction'] = 0
          else:
            bldg['exterior_floor_fraction'] = w / (2. * (w + d)) / (floor_area / (floor_area_choose / 3.))

          bldg['thermal_mass_per_floor_area'] = 3.9 * (0.5 + 1. * np.random.random())
          bldg['interior_exterior_wall_ratio'] = floor_area / (bldg['ceiling_height'] * 2. * (w + d)) - 1. \
            + bldg['window_wall_ratio'] * bldg['exterior_wall_fraction']
          bldg['no_of_doors'] = 0.1  # will round to zero, presumably the exterior doors are treated like windows

          bldg['init_temp'] = 68. + 4. * np.random.random()
          bldg['os_rand'] = bldg['oversize'] * (0.8 + 0.4 * np.random.random())
          bldg['COP_A'] = ConfigDict['cooling_COP']['value'] * (0.8 + 0.4 * np.random.random())

          bldg['adj_lights'] = (0.9 + 0.1 * np.random.random()) * floor_area / 1000.  # randomize 10# then convert W/sf -> kW
          bldg['adj_plugs'] = (0.9 + 0.2 * np.random.random()) * floor_area / 1000.
          bldg['adj_gas'] = (0.9 + 0.2 * np.random.random()) * floor_area / 1000.
          bldg['adj_ext'] = (0.9 + 0.1 * np.random.random()) * floor_area / 1000.
          bldg['adj_occ'] = (0.9 + 0.1 * np.random.random()) * floor_area / 1000.

          bldg['zonename'] = helpers.gld_strict_name (key + '_bldg_' + str(jjj+1) + '_floor_' + str(floor) + '_zone_' + str(zone))
          write_one_commercial_zone (bldg, op)

  elif comm_type == 'BIGBOX':
    bldg['ceiling_height'] = 14.
    bldg['airchange_per_hour'] = 1.5
    bldg['Rroof'] = 19.
    bldg['Rwall'] = 18.3
    bldg['Rfloor'] = 46.
    bldg['Rdoors'] = 3.
    bldg['int_gains'] = 3.6  # W/sf
    bldg['thermal_mass_per_floor_area'] = 1 # TODO
    bldg['exterior_ceiling_fraction'] = 1 # TODO
    bldg['base_schedule'] = 'bigbox'

    num_bigboxes = int(round(nz / 6.))
    for jjj in range(num_bigboxes):
      bldg['skew_value'] = randomize_commercial_skew()
      floor_area_choose = 20000. * (0.5 + 1. * np.random.random())
      floor_area = floor_area_choose / 6.
      bldg['floor_area'] = floor_area
      bldg['thermal_mass_per_floor_area'] = 3.9 * (0.8 + 0.4 * np.random.random())  # +/- 20#
      bldg['exterior_ceiling_fraction'] = 1.
      bldg['aspect_ratio'] = 1.28301275561855
      total_depth = math.sqrt(floor_area_choose / bldg['aspect_ratio'])
      total_width = bldg['aspect_ratio'] * total_depth
      d = total_width / 3.
      w = total_depth / 2.

      for zone in range(1,7):
        if zone == 2 or zone == 5:
          bldg['exterior_wall_fraction'] = d / (2. * (d + w))
          bldg['exterior_floor_fraction'] = (0. + d) / (2. * (total_width + total_depth)) / (floor_area / floor_area_choose)
        else:
          bldg['exterior_wall_fraction'] = 0.5
          bldg['exterior_floor_fraction'] = (w + d) / (2. * (total_width + total_depth)) / (floor_area / floor_area_choose)
        if zone == 2:
          bldg['window_wall_ratio'] = 0.76
        else:
          bldg['window_wall_ratio'] = 0.

        if zone < 4:
          bldg['no_of_doors'] = 0.1  # this will round to 0
        elif zone == 5:
          bldg['no_of_doors'] = 24.
        else:
          bldg['no_of_doors'] = 1.

        bldg['interior_exterior_wall_ratio'] = (floor_area + bldg['no_of_doors'] * 20.) \
          / (bldg['ceiling_height'] * 2. * (w + d)) - 1. + bldg['window_wall_ratio'] * bldg['exterior_wall_fraction']
        bldg['init_temp'] = 68. + 4. * np.random.random()
        bldg['os_rand'] = bldg['oversize'] * (0.8 + 0.4 * np.random.random())
        bldg['COP_A'] = ConfigDict['cooling_COP']['value'] * (0.8 + 0.4 * np.random.random())

        bldg['adj_lights'] = 1.2 * (0.9 + 0.1 * np.random.random()) * floor_area / 1000.  # randomize 10# then convert W/sf -> kW
        bldg['adj_plugs'] = (0.9 + 0.2 * np.random.random()) * floor_area / 1000.
        bldg['adj_gas'] = (0.9 + 0.2 * np.random.random()) * floor_area / 1000.
        bldg['adj_ext'] = (0.9 + 0.1 * np.random.random()) * floor_area / 1000.
        bldg['adj_occ'] = (0.9 + 0.1 * np.random.random()) * floor_area / 1000.

        bldg['zonename'] = helpers.gld_strict_name (key + '_bldg_' + str(jjj+1) + '_zone_' + str(zone))
        write_one_commercial_zone (bldg, op)

  elif comm_type == 'STRIPMALL':
    bldg['ceiling_height'] = 12 # T)D)
    bldg['airchange_per_hour'] = 1.76
    bldg['Rroof'] = 19.
    bldg['Rwall'] = 18.3
    bldg['Rfloor'] = 40.
    bldg['Rdoors'] = 3.
    bldg['int_gains'] = 3.6  # W/sf
    bldg['exterior_ceiling_fraction'] = 1.
    bldg['base_schedule'] = 'stripmall'
    midzone = int (math.floor(nz / 2.) + 1.)
    for zone in range (1, nz+1):
      bldg['skew_value'] = randomize_commercial_skew()
      floor_area_choose = 2400. * (0.7 + 0.6 * np.random.random())
      bldg['thermal_mass_per_floor_area'] = 3.9 * (0.5 + 1. * np.random.random())
      bldg['no_of_doors'] = 1
      if zone == 1 or zone == midzone:
        floor_area = floor_area_choose
        bldg['aspect_ratio'] = 1.5
        bldg['window_wall_ratio'] = 0.05
        bldg['exterior_wall_fraction'] = 0.4
        bldg['exterior_floor_fraction'] = 0.8
        bldg['interior_exterior_wall_ratio'] = -0.05
      else:
        floor_area = floor_area_choose / 2.
        bldg['aspect_ratio'] = 3.0
        bldg['window_wall_ratio'] = 0.03
        if zone == nz:
          bldg['exterior_wall_fraction'] = 0.63
          bldg['exterior_floor_fraction'] = 2.
        else:
          bldg['exterior_wall_fraction'] = 0.25
          bldg['exterior_floor_fraction'] = 0.8
        bldg['interior_exterior_wall_ratio'] = -0.40

      bldg['floor_area'] = floor_area

      bldg['init_temp'] = 68. + 4. * np.random.random()
      bldg['os_rand'] = bldg['oversize'] * (0.8 + 0.4 * np.random.random())
      bldg['COP_A'] = ConfigDict['cooling_COP']['value'] * (0.8 + 0.4 * np.random.random())

      bldg['adj_lights'] = (0.8 + 0.4 * np.random.random()) * floor_area / 1000.
      bldg['adj_plugs'] = (0.8 + 0.4 * np.random.random()) * floor_area / 1000.
      bldg['adj_gas'] = (0.8 + 0.4 * np.random.random()) * floor_area / 1000.
      bldg['adj_ext'] = (0.8 + 0.4 * np.random.random()) * floor_area / 1000.
      bldg['adj_occ'] = (0.8 + 0.4 * np.random.random()) * floor_area / 1000.

      bldg['zonename'] = helpers.gld_strict_name (key + '_zone_' + str(zone))
      write_one_commercial_zone (bldg, op)

  if comm_type == 'ZIPLOAD':
    phsva = 1000.0 * kva / nphs
    print ('object load { // street lights', file=op)
    print ('  name {:s};'.format (key + '_streetlights'), file=op)
    print ('  parent {:s};'.format (mtr), file=op)
    print ('  groupid STREETLIGHTS;', file=op)
    print ('  nominal_voltage {:2f};'.format(vln), file=op)
    print ('  phases {:s};'.format (phases), file=op)
    for phs in ['A', 'B', 'C']:
      if phs in phases:
        print ('  impedance_fraction_{:s} {:f};'.format (phs, ConfigDict['c_z_frac']['value']), file=op)
        print ('  current_fraction_{:s} {:f};'.format (phs, ConfigDict['c_i_frac']['value']), file=op)
        print ('  power_fraction_{:s} {:f};'.format (phs, c_p_frac), file=op)
        print ('  impedance_pf_{:s} {:f};'.format (phs, ConfigDict['c_z_pf']['value']), file=op)
        print ('  current_pf_{:s} {:f};'.format (phs, ConfigDict['c_i_pf']['value']), file=op)
        print ('  power_pf_{:s} {:f};'.format (phs, ConfigDict['c_p_pf']['value']), file=op)
        print ('  base_power_{:s} street_lighting*{:.2f};'.format (phs, ConfigDict['light_scalar_comm']['value'] * phsva), file=op)
    print ('};', file=op)
  else:
    print ('object load { // accumulate zones', file=op)
    print ('  name {:s};'.format (key), file=op)
    print ('  parent {:s};'.format (mtr), file=op)
    print ('  groupid {:s};'.format (comm_type), file=op)
    print ('  nominal_voltage {:2f};'.format(vln), file=op)
    print ('  phases {:s};'.format (phases), file=op)
    print ('};', file=op)

#***************************************************************************************************
#***************************************************************************************************

def write_houses(basenode, op, vnom, bIgnoreThermostatSchedule=True, bWriteService=True, bTriplex=True, setpoint_offset=1.0):
    """Put houses, along with solar panels and batteries, onto a node

    Args:
        basenode (str): GridLAB-D node name
        op (file): open file to write to
        vnom (float): nominal line-to-neutral voltage at basenode
    """
    global ConfigDict

    meter_class = 'triplex_meter'
    node_class = 'triplex_node'
    if bTriplex == False:
        meter_class = 'meter'
        node_class = 'node'

    nhouse = int(ConfigDict['house_nodes']['value'][basenode][0])
    rgn = int(ConfigDict['house_nodes']['value'][basenode][1])
    lg_v_sm = float(ConfigDict['house_nodes']['value'][basenode][2])
    phs = ConfigDict['house_nodes']['value'][basenode][3]
    bldg = ConfigDict['house_nodes']['value'][basenode][4]
    ti = ConfigDict['house_nodes']['value'][basenode][5]
    rgnTable = ConfigDict['rgnThermalPct']['value'][rgn-1]

    if 'A' in phs:
        vstart = str(vnom) + '+0.0j'
    elif 'B' in phs:
        vstart = format(-0.5*vnom,'.2f') + format(-0.866025*vnom,'.2f') + 'j'
    else:
        vstart = format(-0.5*vnom,'.2f') + '+' + format(0.866025*vnom,'.2f') + 'j'

    if ConfigDict['forERCOT']['value'] == "True":
        phs = phs + 'S'
        tpxname = helpers.gld_strict_name (basenode + '_tpx')
        mtrname = helpers.gld_strict_name (basenode + '_mtr')
    elif bWriteService == True:
        print ('object {:s} {{'.format (node_class), file=op)
        print ('  name', basenode + ';', file=op)
        print ('  phases', phs + ';', file=op)
        print ('  nominal_voltage ' + str(vnom) + ';', file=op)
        print ('  voltage_1 ' + vstart + ';', file=op)  # TODO: different attributes for regular node
        print ('  voltage_2 ' + vstart + ';', file=op)
        print ('}', file=op)
    else:
        mtrname = helpers.gld_strict_name (basenode + '_mtr')
    for i in range(nhouse):
        if (ConfigDict["forERCOT"]['value'] == "False") and (bWriteService == True):

            tpxname = helpers.gld_strict_name (basenode + '_tpx_' + str(i+1))
            mtrname = helpers.gld_strict_name (basenode + '_mtr_' + str(i+1))
            print ('object triplex_line {', file=op)
            print ('  name', tpxname + ';', file=op)
            print ('  from', basenode + ';', file=op)
            print ('  to', mtrname + ';', file=op)
            print ('  phases', phs + ';', file=op)
            print ('  length 30;', file=op)
            print ('  configuration', ConfigDict['name_prefix']['value'] + list(ConfigDict['triplex_configurations'].keys())[0] + ';', file=op)
            print ('}', file=op)
            print ('object triplex_meter {', file=op)
            print ('  name', mtrname + ';', file=op)
            print ('  phases', phs + ';', file=op)
            print ('  meter_power_consumption 1+7j;', file=op)
            write_tariff (op)
            print ('  nominal_voltage ' + str(vnom) + ';', file=op)
            print ('  voltage_1 ' + vstart + ';', file=op)
            print ('  voltage_2 ' + vstart + ';', file=op)
            if ConfigDict['metrics_interval']['value'] > 0:
                print ('  object metrics_collector {', file=op)
                print ('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
                print ('  };', file=op)
            print ('}', file=op)
        hsename = helpers.gld_strict_name (basenode + '_hse_' + str(i+1))
        whname = helpers.gld_strict_name (basenode + '_wh_' + str(i+1))
        solname = helpers.gld_strict_name (basenode + '_sol_' + str(i+1))
        batname = helpers.gld_strict_name (basenode + '_bat_' + str(i+1))
        sol_i_name = helpers.gld_strict_name (basenode + '_isol_' + str(i+1))
        bat_i_name = helpers.gld_strict_name (basenode + '_ibat_' + str(i+1))
        sol_m_name = helpers.gld_strict_name (basenode + '_msol_' + str(i+1))
        bat_m_name = helpers.gld_strict_name (basenode + '_mbat_' + str(i+1))
        if ConfigDict['forERCOT']['value'] == "True":
          hse_m_name = mtrname
        else:
          hse_m_name = helpers.gld_strict_name (basenode + '_mhse_' + str(i+1))
          print ('object {:s} {{'.format (meter_class), file=op)
          print ('  name', hse_m_name + ';', file=op)
          print ('  parent', mtrname + ';', file=op)
          print ('  phases', phs + ';', file=op)
          print ('  nominal_voltage ' + str(vnom) + ';', file=op)
          print ('}', file=op)

        fa_base = ConfigDict['rgnFloorArea']['value'][rgn-1][bldg]
        fa_rand = np.random.uniform (0, 1)
        stories = 1
        ceiling_height = 8
        if bldg == 0: # SF homes
            floor_area = fa_base + 0.5 * fa_base * fa_rand * (ti - 3) / 3;
            if np.random.uniform (0, 1) > ConfigDict['rgnOneStory']['value'][rgn-1]:
                stories = 2
            ceiling_height += np.random.randint (0, 2)
        else: # apartment or MH
            floor_area = fa_base + 0.5 * fa_base * (0.5 - fa_rand) # +/- 50%
        floor_area = (1 + lg_v_sm) * floor_area # adjustment depends on whether nhouses rounded up or down
        if floor_area > 4000:
            floor_area = 3800 + fa_rand*200;
        elif floor_area < 300:
            floor_area = 300 + fa_rand*100;

        scalar1 = 324.9/8907 * floor_area**0.442
        scalar2 = 0.8 + 0.4 * np.random.uniform(0,1)
        scalar3 = 0.8 + 0.4 * np.random.uniform(0,1)
        resp_scalar = scalar1 * scalar2
        unresp_scalar = scalar1 * scalar3

        skew_value = ConfigDict['residential_skew_std']['value'] * np.random.randn ()
        if skew_value < -ConfigDict['residential_skew_max']['value']:
            skew_value = -ConfigDict['residential_skew_max']['value']
        elif skew_value > ConfigDict['residential_skew_max']['value']:
            skew_value = ConfigDict['residential_skew_max']['value']

        oversize = ConfigDict['rgnOversizeFactor']['value'][rgn-1] * (0.8 + 0.4 * np.random.uniform(0,1))
        tiProps = selectThermalProperties (bldg, ti)
        # Rceiling(roof), Rwall, Rfloor, WindowLayers, WindowGlass,Glazing,WindowFrame,Rdoor,AirInfil,COPhi,COPlo
        Rroof = tiProps[0] * (0.8 + 0.4 * np.random.uniform(0,1))
        Rwall = tiProps[1] * (0.8 + 0.4 * np.random.uniform(0,1))
        Rfloor = tiProps[2] * (0.8 + 0.4 * np.random.uniform(0,1))
        glazing_layers = int(tiProps[3])
        glass_type = int(tiProps[4])
        glazing_treatment = int(tiProps[5])
        window_frame = int(tiProps[6])
        Rdoor = tiProps[7] * (0.8 + 0.4 * np.random.uniform(0,1))
        airchange = tiProps[8] * (0.8 + 0.4 * np.random.uniform(0,1))
        init_temp = 68 + 4 * np.random.uniform(0,1)
        mass_floor = 2.5 + 1.5 * np.random.uniform(0,1)
        h_COP = c_COP = tiProps[10] + np.random.uniform(0,1) * (tiProps[9] - tiProps[10])

        print ('object house {', file=op)
        print ('  name', hsename + ';', file=op)
        print ('  parent', hse_m_name + ';', file=op)
        print ('  groupid', ConfigDict['bldgTypeName']['value'][bldg] + ';', file=op)
        #print ('  // thermal_integrity_level', ConfigDict['tiName']['value'][ti] + ';', file=op)
        print ('  // thermal_integrity_level', ConfigDict['thermal_integrity_level']['value'][ti] + ';', file=op)
        print ('  schedule_skew', '{:.0f}'.format(skew_value) + ';', file=op)
        print ('  floor_area', '{:.0f}'.format(floor_area) + ';', file=op)
        print ('  number_of_stories', str(stories) + ';', file=op)
        print ('  ceiling_height', str(ceiling_height) + ';', file=op)
        print ('  over_sizing_factor', '{:.1f}'.format(oversize) + ';', file=op)
        print ('  Rroof', '{:.2f}'.format(Rroof) + ';', file=op)
        print ('  Rwall', '{:.2f}'.format(Rwall) + ';', file=op)
        print ('  Rfloor', '{:.2f}'.format(Rfloor) + ';', file=op)
        print ('  glazing_layers', str (glazing_layers) + ';', file=op)
        print ('  glass_type', str (glass_type) + ';', file=op)
        print ('  glazing_treatment', str (glazing_treatment) + ';', file=op)
        print ('  window_frame', str (window_frame) + ';', file=op)
        print ('  Rdoors', '{:.2f}'.format(Rdoor) + ';', file=op)
        print ('  airchange_per_hour', '{:.2f}'.format(airchange) + ';', file=op)
        print ('  cooling_COP', '{:.1f}'.format(c_COP) + ';', file=op)
        print ('  air_temperature', '{:.2f}'.format(init_temp) + ';', file=op)
        print ('  mass_temperature', '{:.2f}'.format(init_temp) + ';', file=op)
        print ('  total_thermal_mass_per_floor_area', '{:.3f}'.format(mass_floor) + ';', file=op)
        print ('  breaker_amps 1000;', file=op)
        print ('  hvac_breaker_rating 1000;', file=op)
        heat_rand = np.random.uniform(0,1)
        cool_rand = np.random.uniform(0,1)
        if heat_rand <= ConfigDict['rgnPenGasHeat']['value'][rgn-1]:
            print ('  heating_system_type GAS;', file=op)
            if cool_rand <= ConfigDict['electric_cooling_percentage']['value']:
                print ('  cooling_system_type ELECTRIC;', file=op)
            else:
                print ('  cooling_system_type NONE;', file=op)
        elif heat_rand <= ConfigDict['rgnPenGasHeat']['value'][rgn-1] + ConfigDict['rgnPenHeatPump']['value'][rgn-1]:
            print ('  heating_system_type HEAT_PUMP;', file=op);
            print ('  heating_COP', '{:.1f}'.format(h_COP) + ';', file=op);
            print ('  cooling_system_type ELECTRIC;', file=op);
            print ('  auxiliary_strategy DEADBAND;', file=op);
            print ('  auxiliary_system_type ELECTRIC;', file=op);
            print ('  motor_model BASIC;', file=op);
            print ('  motor_efficiency AVERAGE;', file=op);
        elif floor_area * ceiling_height > 12000.0: # electric heat not allowed on large homes
            print ('  heating_system_type GAS;', file=op)
            if cool_rand <= ConfigDict['electric_cooling_percentage']['value']:
                print ('  cooling_system_type ELECTRIC;', file=op)
            else:
                print ('  cooling_system_type NONE;', file=op)
        else:
            print ('  heating_system_type RESISTANCE;', file=op)
            if cool_rand <= ConfigDict['electric_cooling_percentage']['value']:
                print ('  cooling_system_type ELECTRIC;', file=op)
                print ('  motor_model BASIC;', file=op);
                print ('  motor_efficiency GOOD;', file=op);
            else:
                print ('  cooling_system_type NONE;', file=op)

        cooling_sch = np.ceil(ConfigDict['coolingScheduleNumber']['value'] * np.random.uniform (0, 1))
        heating_sch = np.ceil(ConfigDict['heatingScheduleNumber']['value'] * np.random.uniform (0, 1))
        # [Bin Prob, NightTimeAvgDiff, HighBinSetting, LowBinSetting]
        cooling_bin, heating_bin = selectSetpointBins (bldg, np.random.uniform (0,1))
        # randomly choose setpoints within bins, and then widen the separation to account for deadband
        cooling_set = cooling_bin[3] + np.random.uniform(0,1) * (cooling_bin[2] - cooling_bin[3]) + setpoint_offset
        heating_set = heating_bin[3] + np.random.uniform(0,1) * (heating_bin[2] - heating_bin[3]) - setpoint_offset
        cooling_diff = 2.0 * cooling_bin[1] * np.random.uniform(0,1)
        heating_diff = 2.0 * heating_bin[1] * np.random.uniform(0,1)
        cooling_scale = np.random.uniform(0.95, 1.05)
        heating_scale = np.random.uniform(0.95, 1.05)
        cooling_str = 'cooling{:.0f}*{:.4f}+{:.2f}'.format(cooling_sch, cooling_scale, cooling_diff)
        heating_str = 'heating{:.0f}*{:.4f}+{:.2f}'.format(heating_sch, heating_scale, heating_diff)
        # default heating and cooling setpoints are 70 and 75 degrees in GridLAB-D
        # we need more separation to assure no overlaps during transactive simulations
        if bIgnoreThermostatSchedule == True:
          print ('  cooling_setpoint 80.0; // ', cooling_str + ';', file=op)
          print ('  heating_setpoint 60.0; // ', heating_str + ';', file=op)
        else:
          print ('  cooling_setpoint {:s};'.format (cooling_str), file=op)
          print ('  heating_setpoint {:s};'.format (heating_str), file=op)

        # heatgain fraction, Zpf, Ipf, Ppf, Z, I, P
        print ('  object ZIPload { // responsive', file=op)
        print ('    schedule_skew', '{:.0f}'.format(skew_value) + ';', file=op)
        print ('    base_power', 'responsive_loads*' + '{:.2f}'.format(resp_scalar) + ';', file=op)
        print ('    heatgain_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['heatgain_fraction']['value'][0]) + ';', file=op)
        print ('    impedance_pf', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['impedance_pf']['value'][1]) + ';', file=op)
        print ('    current_pf', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['current_pf']['value'][2]) + ';', file=op)
        print ('    power_pf', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['power_pf']['value'][3]) + ';', file=op)
        print ('    impedance_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['impedance_fraction']['value'][4]) + ';', file=op)
        print ('    current_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['current_fraction']['value'][5]) + ';', file=op)
        print ('    power_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['power_fraction']['value'][6]) + ';', file=op)
        print ('  };', file=op)
        print ('  object ZIPload { // unresponsive', file=op)
        print ('    schedule_skew', '{:.0f}'.format(skew_value) + ';', file=op)
        print ('    base_power', 'unresponsive_loads*' + '{:.2f}'.format(unresp_scalar) + ';', file=op)
        print ('    heatgain_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['heatgain_fraction']['value'][0]) + ';', file=op)
        print ('    impedance_pf', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['impedance_pf']['value'][1]) + ';', file=op)
        print ('    current_pf', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['current_pf']['value'][2]) + ';', file=op)
        print ('    power_pf', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['power_pf']['value'][3]) + ';', file=op)
        print ('    impedance_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['impedance_fraction']['value'][4]) + ';', file=op)
        print ('    current_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['current_fraction']['value'][5]) + ';', file=op)
        print ('    power_fraction', '{:.2f}'.format(ConfigDict['ZIPload_parameters']['power_fraction']['value'][6]) + ';', file=op)
        print ('  };', file=op)
        if np.random.uniform (0, 1) <= ConfigDict['water_heater_percentage']['value']: # ConfigDict['rgnPenElecWH']['value'][rgn-1]:
          heat_element = 3.0 + 0.5 * np.random.randint (1,6);  # numpy randint (lo, hi) returns lo..(hi-1)
          tank_set = 110 + 16 * np.random.uniform (0, 1);
          therm_dead = 4 + 4 * np.random.uniform (0, 1);
          tank_UA = 2 + 2 * np.random.uniform (0, 1);
          water_sch = np.ceil(ConfigDict['waterHeaterScheduleNumber']['value'] * np.random.uniform (0, 1))
          water_var = 0.95 + np.random.uniform (0, 1) * 0.1 # +/-5% variability
          wh_demand_type = 'large_'
          sizeIncr = np.random.randint (0,3)  # MATLAB randi(imax) returns 1..imax
          sizeProb = np.random.uniform (0, 1);
          if sizeProb <= ConfigDict['rgnWHSize']['value'][rgn-1][0]:
              wh_size = 20 + sizeIncr * 5
              wh_demand_type = 'small_'
          elif sizeProb <= (ConfigDict['rgnWHSize']['value'][rgn-1][0] + ConfigDict['rgnWHSize']['value'][rgn-1][1]):
              wh_size = 30 + sizeIncr * 10
              if floor_area < 2000.0:
                  wh_demand_type = 'small_'
          else:
              if floor_area < 2000.0:
                  wh_size = 30 + sizeIncr * 10
              else:
                  wh_size = 50 + sizeIncr * 10
          wh_demand_str = wh_demand_type + '{:.0f}'.format(water_sch) + '*' + '{:.2f}'.format(water_var)
          wh_skew_value = 3 * ConfigDict['residential_skew_std']['value'] * np.random.randn ()
          if wh_skew_value < -6 * ConfigDict['residential_skew_max']['value']:
              wh_skew_value = -6 * ConfigDict['residential_skew_max']['value']
          elif wh_skew_value > 6 * ConfigDict['residential_skew_max']['value']:
              wh_skew_value = 6 * ConfigDict['residential_skew_max']['value']
          print ('  object waterheater {', file=op)
          print ('    name', whname + ';', file=op)
          print ('    schedule_skew','{:.0f}'.format(wh_skew_value) + ';', file=op)
          print ('    heating_element_capacity','{:.1f}'.format(heat_element), 'kW;', file=op)
          print ('    thermostat_deadband','{:.1f}'.format(therm_dead) + ';', file=op)
          print ('    location INSIDE;', file=op)
          print ('    tank_diameter 1.5;', file=op)
          print ('    tank_UA','{:.1f}'.format(tank_UA) + ';', file=op)
          print ('    water_demand', wh_demand_str + ';', file=op)
          print ('    tank_volume','{:.0f}'.format(wh_size) + ';', file=op)
          if np.random.uniform (0, 1) <= ConfigDict['water_heater_participation']['value']:
              print ('    waterheater_model MULTILAYER;', file=op)
              print ('    discrete_step_size 60.0;', file=op)
              print ('    lower_tank_setpoint','{:.1f}'.format(tank_set - 5.0) + ';', file=op)
              print ('    upper_tank_setpoint','{:.1f}'.format(tank_set + 5.0) + ';', file=op)
              print ('    T_mixing_valve','{:.1f}'.format(tank_set) + ';', file=op)
          else:
              print ('    tank_setpoint','{:.1f}'.format(tank_set) + ';', file=op)
          if ConfigDict['metrics_interval']['value'] > 0:
              print ('    object metrics_collector {', file=op)
              print ('      interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
              print ('    };', file=op)
          print ('  };', file=op)
        if ConfigDict['metrics_interval']['value'] > 0:
            print ('  object metrics_collector {', file=op)
            print ('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
            print ('  };', file=op)
        print ('}', file=op)
        # if PV is allowed, then only single-family houses can buy it, and only the single-family houses with PV will also consider storage
        # if PV is not allowed, then any single-family house may consider storage (if allowed)
        # apartments and mobile homes may always consider storage, but not PV
        bConsiderStorage = True
        if bldg == 0:  # Single-family homes
            if ConfigDict['solar_percentage']['value'] > 0.0:
                bConsiderStorage = False
            if np.random.uniform (0, 1) <= ConfigDict['solar_percentage']['value']:  # some single-family houses have PV
                bConsiderStorage = True
                panel_area = 0.1 * floor_area
                if panel_area < 162:
                    panel_area = 162
                elif panel_area > 270:
                    panel_area = 270
                #inv_power = inv_undersizing * (panel_area/10.7642) * rated_insolation * array_efficiency
                inv_power = ConfigDict['solar']['inv_undersizing']['value'] * (panel_area/10.7642) * ConfigDict['solar']['rated_insolation']['value'] * ConfigDict['solar']['array_efficiency']['value']
                ConfigDict['solar_count']['value'] += 1
                ConfigDict['solar_kw']['value'] += 0.001 * inv_power
                print ('object {:s} {{'.format (meter_class), file=op)
#                print ('object triplex_meter {', file=op)
                print ('  name', sol_m_name + ';', file=op)
                print ('  parent', mtrname + ';', file=op)
                print ('  phases', phs + ';', file=op)
                print ('  nominal_voltage ' + str(vnom) + ';', file=op)
                print ('  object inverter {', file=op)
                print ('    name', sol_i_name + ';', file=op)
                print ('    phases', phs + ';', file=op)
                print ('    generator_status ONLINE;', file=op)
                print ('    inverter_type FOUR_QUADRANT;', file=op)
                print ('    inverter_efficiency 1;', file=op)
                print ('    rated_power','{:.0f}'.format(inv_power) + ';', file=op)
                print ('    power_factor 1.0;', file=op)
                write_solar_inv_settings (op)
                print ('    object solar {', file=op)
                print ('      name', solname + ';', file=op)
                print ('      panel_type SINGLE_CRYSTAL_SILICON;', file=op)
                print ('      efficiency','{:.2f}'.format(ConfigDict['solar']['array_efficiency']['value']) + ';', file=op)
                print ('      area','{:.2f}'.format(panel_area) + ';', file=op)
                print ('    };', file=op)
                if ConfigDict['metrics_interval']['value'] > 0:
                    print ('    object metrics_collector {', file=op)
                    #print ('      interval', str(metrics_interval) + ';', file=op)
                    print ('      interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
                    print ('    };', file=op)
                print ('  };', file=op)
                print ('}', file=op)
        if bConsiderStorage:
            if np.random.uniform (0, 1) <= ConfigDict['storage_percentage']['value']:
                ConfigDict['battery_count']['value'] += 1
                print ('object {:s} {{'.format (meter_class), file=op)
#                print ('object triplex_meter {', file=op)
                print ('  name', bat_m_name + ';', file=op)
                print ('  parent', mtrname + ';', file=op)
                print ('  phases', phs + ';', file=op)
                print ('  nominal_voltage ' + str(vnom) + ';', file=op)
                print ('  object inverter {', file=op)
                print ('    name', bat_i_name + ';', file=op)
                print ('    phases', phs + ';', file=op)
                print ('    generator_status ONLINE;', file=op)
                print ('    generator_mode CONSTANT_PQ;', file=op)
                print ('    inverter_type FOUR_QUADRANT;', file=op)
                print ('    four_quadrant_control_mode', ConfigDict['storage_inv_mode']['value'] + ';', file=op)
                print ('    V_base ${INV_VBASE};', file=op)
                print ('    charge_lockout_time 1;', file=op)
                print ('    discharge_lockout_time 1;', file=op)
                print ('    rated_power 5000;', file=op)
                print ('    max_charge_rate 5000;', file=op)
                print ('    max_discharge_rate 5000;', file=op)
                print ('    sense_object', mtrname + ';', file=op)
                print ('    charge_on_threshold -100;', file=op)
                print ('    charge_off_threshold 0;', file=op)
                print ('    discharge_off_threshold 2000;', file=op)
                print ('    discharge_on_threshold 3000;', file=op)
                print ('    inverter_efficiency 0.97;', file=op)
                print ('    power_factor 1.0;', file=op)
                print ('    object battery { // Tesla Powerwall 2', file=op)
                print ('      name', batname + ';', file=op)
                print ('      use_internal_battery_model true;', file=op)
                print ('      battery_type LI_ION;', file=op)
                print ('      nominal_voltage 480;', file=op)
                print ('      battery_capacity 13500;', file=op)
                print ('      round_trip_efficiency 0.86;', file=op)
                print ('      state_of_charge 0.50;', file=op)
                print ('    };', file=op)
                if ConfigDict['metrics_interval']['value'] > 0:
                    print ('    object metrics_collector {', file=op)
                    print ('      interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
                    print ('    };', file=op)
                print ('  };', file=op)
                print ('}', file=op)

#***************************************************************************************************
#***************************************************************************************************
def write_substation(op, name, phs, vnom, vll):
    """Write the substation swing node, transformer, metrics collector and fncs_msg object

    Args:
        op (file): an open GridLAB-D input file
        name (str): node name of the primary (not transmission) substation bus
        phs (str): primary phasing in the substation
        vnom (float): not used
        vll (float): feeder primary line-to-line voltage
    """
    # if this feeder will be combined with others, need USE_FNCS to appear first as a marker for the substation
    if len(ConfigDict['fncs_case']['value']) > 0:
        #print('#ifdef USE_FNCS', file=op)
        print('#ifdef USE_HELICS',file=op)

        print('object fncs_msg {', file=op)
        if ConfigDict["forERCOT"]['value'] == "True":
            # print ('  name gridlabd' + fncs_case + ';', file=op)
            print('  name gridlabd' + ConfigDict['fncs_case']['value'] + ';', file=op)
        else:
            print('  name gld1;', file=op)
        print('  parent network_node;', file=op)
        print('  configure', ConfigDict['fncs_case']['value'] + '_FNCS_Config.txt;', file=op)
        print('  option "transport:hostname localhost, port 5570";', file=op)
        print('  aggregate_subscriptions true;', file=op)
        print('  aggregate_publications true;', file=op)
        print('}', file=op)
        print('#endif', file=op)
        print('#ifdef USE_HELICS', file=op)
        print('object helics_msg {', file=op)
        print('  configure', ConfigDict['fncs_case']['value'] + '_HELICS_gld_msg.json;', file=op)
        print('}', file=op)
        print('#endif', file=op)
    print('object transformer_configuration {', file=op)
    print('  name substation_xfmr_config;', file=op)
    print('  connect_type WYE_WYE;', file=op)
    print('  install_type PADMOUNT;', file=op)
    print('  primary_voltage', '{:.2f}'.format(ConfigDict['transmissionVoltage']['value']) + ';', file=op)
    print('  secondary_voltage', '{:.2f}'.format(vll) + ';', file=op)
    print('  power_rating', '{:.2f}'.format(ConfigDict['xmfr']['transmissionXfmrMVAbase']['value'] * 1000.0) + ';', file=op)
    print('  resistance', '{:.2f}'.format(0.01 * ConfigDict['xmfr']['transmissionXfmrRpct']['value']) + ';', file=op)
    print('  reactance', '{:.2f}'.format(0.01 * ConfigDict['xmfr']['transmissionXfmrXpct']['value']) + ';', file=op)
    print('  shunt_resistance', '{:.2f}'.format(100.0 / ConfigDict['xmfr']['transmissionXfmrNLLpct']['value']) + ';', file=op)
    print('  shunt_reactance', '{:.2f}'.format(100.0 / ConfigDict['xmfr']['transmissionXfmrImagpct']['value']) + ';', file=op)
    print('}', file=op)
    print('object transformer {', file=op)
    print('  name substation_transformer;', file=op)
    print('  from network_node;', file=op)
    print('  to', name + ';', file=op)
    print('  phases', phs + ';', file=op)
    print('  configuration substation_xfmr_config;', file=op)
    print('}', file=op)
    vsrcln = ConfigDict["transmissionVoltage"]['value'] / math.sqrt(3.0)
    print('object substation {', file=op)
    print('  name network_node;', file=op)
    print('  groupid', ConfigDict['base_feeder_name']['value'] + ';', file=op)
    print('  bustype SWING;', file=op)
    print('  nominal_voltage', '{:.2f}'.format(vsrcln) + ';', file=op)
    print('  positive_sequence_voltage', '{:.2f}'.format(vsrcln) + ';', file=op)
    print('  base_power', '{:.2f}'.format(ConfigDict['xmfr']['transmissionXfmrMVAbase']['value'] * 1000000.0) + ';', file=op)
    print('  power_convergence_value 100.0;', file=op)
    print('  phases', phs + ';', file=op)
    if ConfigDict['metrics_interval']['value'] > 0:
        print('  object metrics_collector {', file=op)
        print('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
        print('  };', file=op)
    print('}', file=op)


# ***************************************************************************************************
# ***************************************************************************************************

# if triplex load, node or meter, the nominal voltage is 120
#   if the name or parent attribute is found in secmtrnode, we look up the nominal voltage there
#   otherwise, the nominal voltage is vprim
# secmtrnode[mtr_node] = [kva_total, phases, vnom]
#   the transformer phasing was not changed, and the transformers were up-sized to the largest phase kva
#   therefore, it should not be necessary to look up kva_total, but phases might have changed N==>S
# if the phasing did change N==>S, we have to prepend triplex_ to the class, write power_1 and voltage_1
# when writing commercial buildings, if load_class is present and == C, skip the instance
def write_voltage_class(model, h, t, op, vprim, vll, secmtrnode):
    """Write GridLAB-D instances that have a primary nominal voltage, i.e., node, meter and load

    Args:
        model (dict): a parsed GridLAB-D model
        h (dict): the object ID hash
        t (str): the GridLAB-D class name to write
        op (file): an open GridLAB-D input file
        vprim (float): the primary nominal line-to-neutral voltage
        vll (float): the primary nominal line-to-line voltage
        secmtrnode (dict): key to [transfomer kva, phasing, nominal voltage] by secondary node name
    """
    if t in model:
        for o in model[t]:
            #            if 'load_class' in model[t][o]:
            #                if model[t][o]['load_class'] == 'C':
            #                    continue
            name = o  # model[t][o]['name']
            phs = model[t][o]['phases']
            vnom = vprim
            if 'bustype' in model[t][o]:
                if model[t][o]['bustype'] == 'SWING':
                    write_substation(op, name, phs, vnom, vll)
            parent = ''
            prefix = ''
            if str.find(phs, 'S') >= 0:
                bHadS = True
            else:
                bHadS = False
            if str.find(name, '_tn_') >= 0 or str.find(name, '_tm_') >= 0:
                vnom = 120.0
            if name in secmtrnode:
                vnom = secmtrnode[name][2]
                phs = secmtrnode[name][1]
            if 'parent' in model[t][o]:
                parent = model[t][o]['parent']
                if parent in secmtrnode:
                    vnom = secmtrnode[parent][2]
                    phs = secmtrnode[parent][1]
            if str.find(phs, 'S') >= 0:
                bHaveS = True
            else:
                bHaveS = False
            if bHaveS == True and bHadS == False:
                prefix = 'triplex_'
            print('object ' + prefix + t + ' {', file=op)
            if len(parent) > 0:
                print('  parent ' + parent + ';', file=op)
            print('  name ' + name + ';', file=op)
            if 'groupid' in model[t][o]:
                print('  groupid ' + model[t][o]['groupid'] + ';', file=op)
            if 'bustype' in model[t][o]:  # already moved the SWING bus behind substation transformer
                if model[t][o]['bustype'] != 'SWING':
                    print('  bustype ' + model[t][o]['bustype'] + ';', file=op)
            print('  phases ' + phs + ';', file=op)
            print('  nominal_voltage ' + str(vnom) + ';', file=op)
            if 'load_class' in model[t][o]:
                print('  load_class ' + model[t][o]['load_class'] + ';', file=op)
            if 'constant_power_A' in model[t][o]:
                if bHaveS == True:
                    print('  power_1 ' + model[t][o]['constant_power_A'] + ';', file=op)
                else:
                    print('  constant_power_A ' + model[t][o]['constant_power_A'] + ';', file=op)
            if 'constant_power_B' in model[t][o]:
                if bHaveS == True:
                    print('  power_1 ' + model[t][o]['constant_power_B'] + ';', file=op)
                else:
                    print('  constant_power_B ' + model[t][o]['constant_power_B'] + ';', file=op)
            if 'constant_power_C' in model[t][o]:
                if bHaveS == True:
                    print('  power_1 ' + model[t][o]['constant_power_C'] + ';', file=op)
                else:
                    print('  constant_power_C ' + model[t][o]['constant_power_C'] + ';', file=op)
            if 'power_1' in model[t][o]:
                print('  power_1 ' + model[t][o]['power_1'] + ';', file=op)
            if 'power_2' in model[t][o]:
                print('  power_2 ' + model[t][o]['power_2'] + ';', file=op)
            if 'power_12' in model[t][o]:
                print('  power_12 ' + model[t][o]['power_12'] + ';', file=op)
            vstarta = str(vnom) + '+0.0j'
            vstartb = format(-0.5 * vnom, '.2f') + format(-0.866025 * vnom, '.2f') + 'j'
            vstartc = format(-0.5 * vnom, '.2f') + '+' + format(0.866025 * vnom, '.2f') + 'j'
            if 'voltage_A' in model[t][o]:
                if bHaveS == True:
                    print('  voltage_1 ' + vstarta + ';', file=op)
                    print('  voltage_2 ' + vstarta + ';', file=op)
                else:
                    print('  voltage_A ' + vstarta + ';', file=op)
            if 'voltage_B' in model[t][o]:
                if bHaveS == True:
                    print('  voltage_1 ' + vstartb + ';', file=op)
                    print('  voltage_2 ' + vstartb + ';', file=op)
                else:
                    print('  voltage_B ' + vstartb + ';', file=op)
            if 'voltage_C' in model[t][o]:
                if bHaveS == True:
                    print('  voltage_1 ' + vstartc + ';', file=op)
                    print('  voltage_2 ' + vstartc + ';', file=op)
                else:
                    print('  voltage_C ' + vstartc + ';', file=op)
            if 'power_1' in model[t][o]:
                print('  power_1 ' + model[t][o]['power_1'] + ';', file=op)
            if 'power_2' in model[t][o]:
                print('  power_2 ' + model[t][o]['power_2'] + ';', file=op)
            if 'voltage_1' in model[t][o]:
                if str.find(phs, 'A') >= 0:
                    print('  voltage_1 ' + vstarta + ';', file=op)
                    print('  voltage_2 ' + vstarta + ';', file=op)
                if str.find(phs, 'B') >= 0:
                    print('  voltage_1 ' + vstartb + ';', file=op)
                    print('  voltage_2 ' + vstartb + ';', file=op)
                if str.find(phs, 'C') >= 0:
                    print('  voltage_1 ' + vstartc + ';', file=op)
                    print('  voltage_2 ' + vstartc + ';', file=op)
            if name in extra_billing_meters:
                write_tariff(op)
                if ConfigDict['metrics_interval']['value'] > 0:
                    print('  object metrics_collector {', file=op)
                    print('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
                    print('  };', file=op)
            print('}', file=op)


# ***************************************************************************************************
# ***************************************************************************************************

def write_xfmr_config(key, phs, kvat, vnom, vsec, install_type, vprimll, vprimln, op):
    """Write a transformer_configuration

    Args:
        key (str): name of the configuration
        phs (str): primary phasing
        kvat (float): transformer rating in kVA
        vnom (float): primary voltage rating, not used any longer (see vprimll and vprimln)
        vsec (float): secondary voltage rating, should be line-to-neutral for single-phase or line-to-line for three-phase
        install_type (str): should be VAULT, PADMOUNT or POLETOP
        vprimll (float): primary line-to-line voltage, used for three-phase transformers
        vprimln (float): primary line-to-neutral voltage, used for single-phase transformers
        op (file): an open GridLAB-D input file
    """
    print('object transformer_configuration {', file=op)
    print('  name ' + ConfigDict['name_prefix']['value'] + key + ';', file=op)
    print('  power_rating ' + format(kvat, '.2f') + ';', file=op)
    kvaphase = kvat
    if 'XF2' in key:
        kvaphase /= 2.0
    if 'XF3' in key:
        kvaphase /= 3.0
    if 'A' in phs:
        print('  powerA_rating ' + format(kvaphase, '.2f') + ';', file=op)
    else:
        print('  powerA_rating 0.0;', file=op)
    if 'B' in phs:
        print('  powerB_rating ' + format(kvaphase, '.2f') + ';', file=op)
    else:
        print('  powerB_rating 0.0;', file=op)
    if 'C' in phs:
        print('  powerC_rating ' + format(kvaphase, '.2f') + ';', file=op)
    else:
        print('  powerC_rating 0.0;', file=op)
    print('  install_type ' + install_type + ';', file=op)
    if 'S' in phs:
        row = Find1PhaseXfmr(kvat)
        print('  connect_type SINGLE_PHASE_CENTER_TAPPED;', file=op)
        print('  primary_voltage ' + str(vprimln) + ';', file=op)
        print('  secondary_voltage ' + format(vsec, '.1f') + ';', file=op)
        print('  resistance ' + format(row[1] * 0.5, '.5f') + ';', file=op)
        print('  resistance1 ' + format(row[1], '.5f') + ';', file=op)
        print('  resistance2 ' + format(row[1], '.5f') + ';', file=op)
        print('  reactance ' + format(row[2] * 0.8, '.5f') + ';', file=op)
        print('  reactance1 ' + format(row[2] * 0.4, '.5f') + ';', file=op)
        print('  reactance2 ' + format(row[2] * 0.4, '.5f') + ';', file=op)
        print('  shunt_resistance ' + format(1.0 / row[3], '.2f') + ';', file=op)
        print('  shunt_reactance ' + format(1.0 / row[4], '.2f') + ';', file=op)
    else:
        row = Find3PhaseXfmr(kvat)
        print('  connect_type WYE_WYE;', file=op)
        print('  primary_voltage ' + str(vprimll) + ';', file=op)
        print('  secondary_voltage ' + format(vsec, '.1f') + ';', file=op)
        print('  resistance ' + format(row[1], '.5f') + ';', file=op)
        print('  reactance ' + format(row[2], '.5f') + ';', file=op)
        print('  shunt_resistance ' + format(1.0 / row[3], '.2f') + ';', file=op)
        print('  shunt_reactance ' + format(1.0 / row[4], '.2f') + ';', file=op)
    print('}', file=op)


# ***************************************************************************************************
# ***************************************************************************************************

def ProcessTaxonomyFeeder(outname, rootname, vll, vln, avghouse, avgcommercial):
    """Parse and re-populate one backbone feeder, usually but not necessarily one of the PNNL taxonomy feeders

    This function:

        * reads and parses the backbone model from *rootname.glm*
        * replaces loads with houses and DER
        * upgrades transformers and fuses as needed, based on a radial graph analysis
        * writes the repopulated feeder to *outname.glm*

    Args:
        outname (str): the output feeder model name
        rootname (str): the input (usually taxonomy) feeder model name
        vll (float): the feeder primary line-to-line voltage
        vln (float): the feeder primary line-to-neutral voltage
        avghouse (float): the average house load in kVA
        avgcommercial (float): the average commercial load in kVA, not used
    """
    global ConfigDict
    ConfigDict['solar_count']['value'] = 0
    ConfigDict['solar_kw']['value'] = 0
    ConfigDict['battery_count']['value'] = 0

    ConfigDict['base_feeder_name']['value'] = rootname
    fname = ConfigDict['glmpath']['value'] + rootname + '.glm'
    print('Populating From:', fname)
    rgn = 0
    if 'R1' in rootname:
        rgn = 1
    elif 'R2' in rootname:
        rgn = 2
    elif 'R3' in rootname:
        rgn = 3
    elif 'R4' in rootname:
        rgn = 4
    elif 'R5' in rootname:
        rgn = 5
    print('using', ConfigDict['solar_percentage']['value'], 'solar and', ConfigDict['storage_percentage']['value'],
          'storage penetration')
    if ConfigDict['electric_cooling_percentage']['value'] <= 0.0:
        ConfigDict['electric_cooling_percentage']['value'] = ConfigDict['rgnPenElecCool']['value'][rgn - 1]
        print('using regional default', ConfigDict['electric_cooling_percentage']['value'],
              'air conditioning penetration')
    else:
        print('using', ConfigDict['electric_cooling_percentage']['value'],
              'air conditioning penetration from JSON config')
    if ConfigDict['water_heater_percentage']['value'] <= 0.0:
        ConfigDict['water_heater_percentage']['value'] = ConfigDict['rgnPenElecWH']['value'][rgn - 1]
        print('using regional default', ConfigDict['water_heater_percentage']['value'], 'water heater penetration')
    else:
        print('using', ConfigDict['water_heater_percentage']['value'], 'water heater penetration from JSON config')
    if os.path.isfile(fname):
        ip = open(fname, 'r')
        lines = []
        line = ip.readline()
        while line != '':
            while re.match('\s*//', line) or re.match('\s+$', line):
                # skip comments and white space
                line = ip.readline()
            lines.append(line.rstrip())
            line = ip.readline()
        ip.close()

        op = open(ConfigDict['outpath']['value'] + ConfigDict['casefiles']['outname'] + '.glm', 'w')
        print('###### Writing to', ConfigDict['outpath']['value'] + outname + '.glm')
        octr = 0;
        model = {}
        h = {}  # OID hash
        itr = iter(lines)
        for line in itr:
            if re.search('object', line):
                line, octr = obj(None, model, line, itr, h, octr)
            else:  # should be the pre-amble, need to replace timestamp and stoptime
                if 'timestamp' in line:
#                    print('  timestamp \'' + ConfigDict['starttime']['value'] + '\';', file=op)
                    print('  timestamp \'' + ConfigDict['simtime']['starttime'] + '\';', file=op)
                elif 'stoptime' in line:
#                    print('  stoptime \'' + ConfigDict['endtime']['value'] + '\';', file=op)
                    print('  stoptime \'' + ConfigDict['simtime']['endtime'] + '\';', file=op)
                else:
                    print(line, file=op)

        # apply the nameing prefix if necessary
        # if len(name_prefix) > 0:
        if len(ConfigDict['name_prefix']['value']) > 0:
            for t in model:
                for o in model[t]:
                    elem = model[t][o]
                    for tok in ['name', 'parent', 'from', 'to', 'configuration', 'spacing',
                                'conductor_1', 'conductor_2', 'conductor_N',
                                'conductor_A', 'conductor_B', 'conductor_C']:
                        if tok in elem:
                            elem[tok] = ConfigDict['name_prefix']['value'] + elem[tok]

        #        log_model (model, h)

        # construct a graph of the model, starting with known links
        G = nx.Graph()
        for t in model:
            if is_edge_class(t):
                for o in model[t]:
                    n1 = model[t][o]['from']
                    n2 = model[t][o]['to']
                    G.add_edge(n1, n2, eclass=t, ename=o, edata=model[t][o])

        # add the parent-child node links
        for t in model:
            if is_node_class(t):
                for o in model[t]:
                    if 'parent' in model[t][o]:
                        p = model[t][o]['parent']
                        G.add_edge(o, p, eclass='parent', ename=o, edata={})

        # now we backfill node attributes
        for t in model:
            if is_node_class(t):
                for o in model[t]:
                    if o in G.nodes():
                        G.nodes()[o]['nclass'] = t
                        G.nodes()[o]['ndata'] = model[t][o]
                    else:
                        print('orphaned node', t, o)

        swing_node = ''
        for n1, data in G.nodes(data=True):
            if 'nclass' in data:
                if 'bustype' in data['ndata']:
                    if data['ndata']['bustype'] == 'SWING':
                        swing_node = n1

        sub_graphs = nx.connected_components(G)
        seg_loads = {}  # [name][kva, phases]
        total_kva = 0.0
        for n1, data in G.nodes(data=True):
            if 'ndata' in data:
                kva = accumulate_load_kva(data['ndata'])
                # need to account for large-building loads added through transformer connections
                if n1 == ConfigDict['Eplus']['Eplus_Bus']['value']:
                    kva += ConfigDict['Eplus']['Eplus_kVA']['value']
                if kva > 0:
                    total_kva += kva
                    nodes = nx.shortest_path(G, n1, swing_node)
                    edges = zip(nodes[0:], nodes[1:])
                    for u, v in edges:
                        eclass = G[u][v]['eclass']
                        if is_edge_class(eclass):
                            ename = G[u][v]['ename']
                            if ename not in seg_loads:
                                seg_loads[ename] = [0.0, '']
                            seg_loads[ename][0] += kva
                            seg_loads[ename][1] = union_of_phases(seg_loads[ename][1], data['ndata']['phases'])

        print('  swing node', swing_node, 'with', len(list(sub_graphs)), 'subgraphs and',
              '{:.2f}'.format(total_kva), 'total kva')

        # preparatory items for TESP
        print('module climate;', file=op)
        print('module generators;', file=op)
        print('module connection;', file=op)
        print('module residential {', file=op)
        print('  implicit_enduses NONE;', file=op)
        print('};', file=op)
        print('#include "' + ConfigDict['supportpath']['value'] + 'appliance_schedules.glm";', file=op)
        print('#include "' + ConfigDict['supportpath']['value'] + 'water_and_setpoint_schedule_v5.glm";', file=op)
        print('#include "' + ConfigDict['supportpath']['value'] + 'commercial_schedules.glm";', file=op)
        print('#set minimum_timestep=' + str(ConfigDict['timestep']['value']) + ';', file=op)
        print('#set relax_naming_rules=1;', file=op)
        print('#set warn=0;', file=op)
        if ConfigDict['metrics_interval']['value'] > 0:
            print('object metrics_collector_writer {', file=op)
            # print ('  interval', str(metrics_interval) + ';', file=op)
            print('  interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
            print('  interim 43200;', file=op)  # TODO - make this a configuration parameter
            if ConfigDict["forERCOT"]['value'] == "True":
                print('  // filename ${METRICS_FILE};', file=op)
                print('  filename ' + outname + '_metrics.json;', file=op)
            else:
                print('  filename ${METRICS_FILE};', file=op)
                print('  // filename ' + outname + '_metrics.json;', file=op)
            print('};', file=op)
        print('object climate {', file=op)
        print('  name localWeather;', file=op)


        #print('  // tmyfile "' + ConfigDict['weatherpath']['value'] + ConfigDict['weather_file']['value'] + '";',
        #      file=op)
        print('  // tmyfile "' + ConfigDict['climate']['weatherpath']['value'] + ConfigDict['climate']['weather_file']['value'] + '";',
              file=op)
        #print('  // agent name', ConfigDict['weatherName']['value'], file=op)
        print('  // agent name', ConfigDict['climate']['weatherName']['value'], file=op)
        #print('  interpolate QUADRATIC;', file=op)
        print('  interpolate ' + ConfigDict['climate']['interpolate'] +';', file=op)



        print('  latitude', str(ConfigDict['latitude']['value']) + ';', file=op)
        print('  longitude', str(ConfigDict['longitude']['value']) + ';', file=op)
        print('  // altitude', str(ConfigDict['altitude']['value']) + ';', file=op)
        print('  tz_meridian', str(ConfigDict['tz_meridian']['value']) + ';', file=op)
        print('};', file=op)
        #        print ('// taxonomy_base_feeder', rootname, file=op)
        #        print ('// region_name', ConfigDict['rgnName']['value'][rgn-1], file=op)
        if ConfigDict['solar_percentage']['value'] > 0.0:
            print('// default IEEE 1547-2018 settings for Category B', file=op)
            print('#define INV_VBASE=240.0', file=op)
            print('#define INV_V1=0.92', file=op)
            print('#define INV_V2=0.98', file=op)
            print('#define INV_V3=1.02', file=op)
            print('#define INV_V4=1.08', file=op)
            print('#define INV_Q1=0.44', file=op)
            print('#define INV_Q2=0.00', file=op)
            print('#define INV_Q3=0.00', file=op)
            print('#define INV_Q4=-0.44', file=op)
            print('#define INV_VIN=200.0', file=op)
            print('#define INV_IIN=32.5', file=op)
            print('#define INV_VVLOCKOUT=300.0', file=op)
            print('#define INV_VW_V1=1.05 // 1.05833', file=op)
            print('#define INV_VW_V2=1.10', file=op)
            print('#define INV_VW_P1=1.0', file=op)
            print('#define INV_VW_P2=0.0', file=op)
        # write the optional volt_dump and curr_dump for validation
        print('#ifdef WANT_VI_DUMP', file=op)
        print('object voltdump {', file=op)
        print('  filename Voltage_Dump_' + outname + '.csv;', file=op)
        print('  mode polar;', file=op)
        print('}', file=op)
        print('object currdump {', file=op)
        print('  filename Current_Dump_' + outname + '.csv;', file=op)
        print('  mode polar;', file=op)
        print('}', file=op)
        print('#endif // &&& end of common section for combining TESP cases', file=op)
        print('// solar inverter mode on this feeder', file=op)
        print(
            '#define ' + ConfigDict['name_prefix']['value'] + 'INVERTER_MODE=' + ConfigDict['solar_inv_mode']['value'],
            file=op)

        # NEW STRATEGY - loop through transformer instances and assign a standard size based on the downstream load
        #              - change the referenced transformer_configuration attributes
        #              - write the standard transformer_configuration instances we actually need
        xfused = {}  # ID, phases, total kva, vnom (LN), vsec, poletop/padmount
        secnode = {}  # Node, st, phases, vnom
        t = 'transformer'
        if t not in model:
            model[t] = {}
        for o in model[t]:
            seg_kva = seg_loads[o][0]
            seg_phs = seg_loads[o][1]
            nphs = 0
            if 'A' in seg_phs:
                nphs += 1
            if 'B' in seg_phs:
                nphs += 1
            if 'C' in seg_phs:
                nphs += 1
            if nphs > 1:
                kvat = Find3PhaseXfmrKva(seg_kva)
            else:
                kvat = Find1PhaseXfmrKva(seg_kva)
            if 'S' in seg_phs:
                vnom = 120.0
                vsec = 120.0
            else:
                if 'N' not in seg_phs:
                    seg_phs += 'N'
                if kvat > ConfigDict['max208kva']['value']:
                    vsec = 480.0
                    vnom = 277.0
                else:
                    vsec = 208.0
                    vnom = 120.0

            secnode[model[t][o]['to']] = [kvat, seg_phs, vnom]

            old_key = h[model[t][o]['configuration']]
            install_type = model['transformer_configuration'][old_key]['install_type']

            raw_key = 'XF' + str(nphs) + '_' + install_type + '_' + seg_phs + '_' + str(kvat)
            key = raw_key.replace('.', 'p')

            model[t][o]['configuration'] = ConfigDict['name_prefix']['value'] + key
            model[t][o]['phases'] = seg_phs
            if key not in xfused:
                xfused[key] = [seg_phs, kvat, vnom, vsec, install_type]

        for key in xfused:
            write_xfmr_config(key, xfused[key][0], xfused[key][1], xfused[key][2], xfused[key][3],
                              xfused[key][4], vll, vln, op)

        t = 'capacitor'
        if t in model:
            for o in model[t]:
                model[t][o]['nominal_voltage'] = str(int(vln))
                model[t][o]['cap_nominal_voltage'] = str(int(vln))

        t = 'fuse'
        if t not in model:
            model[t] = {}
        for o in model[t]:
            if o in seg_loads:
                seg_kva = seg_loads[o][0]
                seg_phs = seg_loads[o][1]
                nphs = 0
                if 'A' in seg_phs:
                    nphs += 1
                if 'B' in seg_phs:
                    nphs += 1
                if 'C' in seg_phs:
                    nphs += 1
                if nphs == 3:
                    amps = 1000.0 * seg_kva / math.sqrt(3.0) / vll
                elif nphs == 2:
                    amps = 1000.0 * seg_kva / 2.0 / vln
                else:
                    amps = 1000.0 * seg_kva / vln
                model[t][o]['current_limit'] = str(FindFuseLimit(amps))

        write_local_triplex_configurations(op)

        write_config_class(model, h, 'regulator_configuration', op)
        write_config_class(model, h, 'overhead_line_conductor', op)
        write_config_class(model, h, 'line_spacing', op)
        write_config_class(model, h, 'line_configuration', op)
        write_config_class(model, h, 'triplex_line_conductor', op)
        write_config_class(model, h, 'triplex_line_configuration', op)
        write_config_class(model, h, 'underground_line_conductor', op)

        write_link_class(model, h, 'fuse', seg_loads, op)
        write_link_class(model, h, 'switch', seg_loads, op)
        write_link_class(model, h, 'recloser', seg_loads, op)
        write_link_class(model, h, 'sectionalizer', seg_loads, op)

        write_link_class(model, h, 'overhead_line', seg_loads, op)
        write_link_class(model, h, 'underground_line', seg_loads, op)
        write_link_class(model, h, 'series_reactor', seg_loads, op)

        write_link_class(model, h, 'regulator', seg_loads, op, want_metrics=True)
        write_link_class(model, h, 'transformer', seg_loads, op)
        write_link_class(model, h, 'capacitor', seg_loads, op, want_metrics=True)

        if ConfigDict["forERCOT"]['value'] == "True":
            replace_commercial_loads(model, h, 'load', 0.001 * avgcommercial)
            #            connect_ercot_commercial (op)
            identify_ercot_houses(model, h, 'load', 0.001 * avghouse, rgn)
            connect_ercot_houses(model, h, op, vln, 120.0)
            for key in ConfigDict['house_nodes']['value']:
                write_houses(key, op, 120.0)
            for key in ConfigDict['small_nodes']['value']:
                write_ercot_small_loads(key, op, vln)
            for key in ConfigDict['comm_loads']['value']:
                write_commercial_loads(rgn, key, op)
        else:
            replace_commercial_loads(model, h, 'load', 0.001 * avgcommercial)
            identify_xfmr_houses(model, h, 'transformer', seg_loads, 0.001 * avghouse, rgn)
            for key in ConfigDict['house_nodes']['value']:
                write_houses(key, op, 120.0)
            for key in ConfigDict['small_nodes']['value']:
                write_small_loads(key, op, 120.0)
            for key in ConfigDict['comm_loads']['value']:
                write_commercial_loads(rgn, key, op)

        write_voltage_class(model, h, 'node', op, vln, vll, secnode)
        write_voltage_class(model, h, 'meter', op, vln, vll, secnode)
        if ConfigDict["forERCOT"]['value'] == "False":
            write_voltage_class(model, h, 'load', op, vln, vll, secnode)
        if len(ConfigDict['Eplus']['Eplus_Bus']['value']) > 0 and ConfigDict['Eplus']['Eplus_Volts']['value'] > 0.0 and \
                ConfigDict['Eplus']['Eplus_kVA']['value'] > 0.0:
            print('////////// EnergyPlus large-building load ///////////////', file=op)
            row = Find3PhaseXfmr(ConfigDict['Eplus']['Eplus_kVA']['value'])
            actual_kva = row[0]
            watts_per_phase = 1000.0 * actual_kva / 3.0
            Eplus_vln = ConfigDict['Eplus']['Eplus_Volts']['value'] / math.sqrt(3.0)
            vstarta = format(Eplus_vln, '.2f') + '+0.0j'
            vstartb = format(-0.5 * Eplus_vln, '.2f') + format(-0.866025 * Eplus_vln, '.2f') + 'j'
            vstartc = format(-0.5 * Eplus_vln, '.2f') + '+' + format(0.866025 * Eplus_vln, '.2f') + 'j'
            print('object transformer_configuration {', file=op)
            print('  name ' + ConfigDict['name_prefix']['value'] + 'Eplus_transformer_configuration;', file=op)
            print('  connect_type WYE_WYE;', file=op)
            print('  install_type PADMOUNT;', file=op)
            print('  power_rating', str(actual_kva) + ';', file=op)
            print('  primary_voltage ' + str(vll) + ';', file=op)
            print('  secondary_voltage ' + format(ConfigDict['Eplus']['Eplus_Volts']['value'], '.1f') + ';', file=op)
            print('  resistance ' + format(row[1], '.5f') + ';', file=op)
            print('  reactance ' + format(row[2], '.5f') + ';', file=op)
            print('  shunt_resistance ' + format(1.0 / row[3], '.2f') + ';', file=op)
            print('  shunt_reactance ' + format(1.0 / row[4], '.2f') + ';', file=op)
            print('}', file=op)
            print('object transformer {', file=op)
            print('  name ' + ConfigDict['name_prefix']['value'] + 'Eplus_transformer;', file=op)
            print('  phases ABCN;', file=op)
            print('  from', ConfigDict['name_prefix']['value'] + ConfigDict['Eplus']['Eplus_Bus']['value'] + ';', file=op)
            print('  to', ConfigDict['name_prefix']['value'] + 'Eplus_meter;', file=op)
            print('  configuration ' + ConfigDict['name_prefix']['value'] + 'Eplus_transformer_configuration;', file=op)
            print('}', file=op)
            print('object meter {', file=op)
            print('  name ' + ConfigDict['name_prefix']['value'] + 'Eplus_meter;', file=op)
            print('  phases ABCN;', file=op)
            print('  meter_power_consumption 1+15j;', file=op)
            print('  nominal_voltage', '{:.4f}'.format(Eplus_vln) + ';', file=op)
            print('  voltage_A ' + vstarta + ';', file=op)
            print('  voltage_B ' + vstartb + ';', file=op)
            print('  voltage_C ' + vstartc + ';', file=op)
            write_tariff(op)
            if ConfigDict['metrics_interval']['value'] > 0:
                print('  object metrics_collector {', file=op)
                print('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=op)
                print('  };', file=op)
            print('}', file=op)
            print('object load {', file=op)
            print('  name ' + ConfigDict['name_prefix']['value'] + 'Eplus_load;', file=op)
            print('  parent ' + ConfigDict['name_prefix']['value'] + 'Eplus_meter;', file=op)
            print('  phases ABCN;', file=op)
            print('  nominal_voltage', '{:.4f}'.format(Eplus_vln) + ';', file=op)
            print('  voltage_A ' + vstarta + ';', file=op)
            print('  voltage_B ' + vstartb + ';', file=op)
            print('  voltage_C ' + vstartc + ';', file=op)
            print('  constant_power_A', '{:.1f}'.format(watts_per_phase) + ';', file=op)
            print('  constant_power_B', '{:.1f}'.format(watts_per_phase) + ';', file=op)
            print('  constant_power_C', '{:.1f}'.format(watts_per_phase) + ';', file=op)
            print('}', file=op)

        print('cooling bins unused', ConfigDict['cooling_bins']['value'])
        print('heating bins unused', ConfigDict['heating_bins']['value'])
        print(ConfigDict['solar_count']['value'], 'pv totaling', '{:.1f}'.format(ConfigDict['solar_kw']['value']), 'kw with',
              ConfigDict['battery_count']['value'], 'batteries')

        op.close()


# ***************************************************************************************************
# ***************************************************************************************************

def write_node_houses(fp, node, region, xfkva, phs, nh=None, loadkw=None, house_avg_kw=None, secondary_ft=None,
                      storage_fraction=0.0, solar_fraction=0.0, electric_cooling_fraction=0.5,
                      node_metrics_interval=None, random_seed=False):
    """Writes GridLAB-D houses to a primary load point.

    One aggregate service transformer is included, plus an optional aggregate secondary service drop. Each house
    has a separate meter or triplex_meter, each with a common parent, either a node or triplex_node on either the
    transformer secondary, or the end of the service drop. The houses may be written per phase, i.e., unbalanced load,
    or as a balanced three-phase load. The houses should be #included into a master GridLAB-D file. Before using this
    function, call write_node_house_configs once, and only once, for each combination xfkva/phs that will be used.

    Args:
        fp (file): Previously opened text file for writing; the caller closes it.
        node (str): the GridLAB-D primary node name
        region (int): the taxonomy region for housing population, 1..6
        xfkva (float): the total transformer size to serve expected load; make this big enough to avoid overloads
        phs (str): 'ABC' for three-phase balanced distribution, 'AS', 'BS', or 'CS' for single-phase triplex
        nh (int): directly specify the number of houses; an alternative to loadkw and house_avg_kw
        loadkw (float): total load kW that the houses will represent; with house_avg_kw, an alternative to nh
        house_avg_kw (float): average house load in kW; with loadkw, an alternative to nh
        secondary_ft (float): if not None, the length of adequately sized secondary circuit from transformer to the meters
        electric_cooling_fraction (float): fraction of houses to have air conditioners
        solar_fraction (float): fraction of houses to have rooftop solar panels
        storage_fraction (float): fraction of houses with solar panels that also have residential storage systems
        node_metrics_interval (int): if not None, the metrics collection interval in seconds for houses, meters, solar and storage at this node
        random_seed (boolean): if True, reseed each function call. Default value False provides repeatability of output.
    """
    global ConfigDict
    ConfigDict['house_nodes']['value'] = {}
    if not random_seed:
        np.random.seed(0)
    bTriplex = False
    if 'S' in phs:
        bTriplex = True
    ConfigDict['storage_percentage']['value'] = storage_fraction
    ConfigDict['solar_percentage']['value'] = solar_fraction
    ConfigDict['electric_cooling_percentage']['value'] = electric_cooling_fraction
    lg_v_sm = 0.0
    vnom = 120.0
    if node_metrics_interval is not None:
        ConfigDict['metrics_interval']['value'] = node_metrics_interval
    else:
        ConfigDict['metrics_interval']['value'] = 0
    if nh is not None:
        nhouse = nh
    else:
        nhouse = int((loadkw / house_avg_kw) + 0.5)
        if nhouse > 0:
            lg_v_sm = loadkw / house_avg_kw - nhouse  # >0 if we rounded down the number of houses
    bldg, ti = selectResidentialBuilding(ConfigDict['rgnThermalPct']['value'][region - 1],
                                         np.random.uniform(0, 1))  # TODO - these will all be identical!
    if nhouse > 0:
        # write the transformer and one billing meter at the house, with optional secondary circuit
        if bTriplex:
            xfkey = 'XF{:s}_{:d}'.format(phs[0], int(xfkva))
            linekey = 'tpx_cfg_{:d}'.format(int(xfkva))
            meter_class = 'triplex_meter'
            line_class = 'triplex_line'
        else:
            xfkey = 'XF3_{:d}'.format(int(xfkva))
            linekey = 'quad_cfg_{:d}'.format(int(xfkva))
            meter_class = 'meter'
            line_class = 'overhead_line'
        if secondary_ft is None:
            xfmr_meter = '{:s}_mtr'.format(node)  # same as the house meter
        else:
            xfmr_meter = '{:s}_xfmtr'.format(node)  # needs its own secondary meter
        if (ConfigDict['solar_percentage']['value'] > 0.0) or (ConfigDict['storage_percentage']['value']) > 0.0:
            if bTriplex:
                print('// inverter base voltage for volt-var functions, on triplex circuit', file=fp)
                print('#define INV_VBASE=240.0', file=fp)
            else:
                print('// inverter base voltage for volt-var functions, on 208-V three-phase circuit', file=fp)
                print('#define INV_VBASE=208.0', file=fp)
        print('object transformer {', file=fp)
        print('  name {:s}_xfmr;'.format(node), file=fp)
        print('  phases {:s};'.format(phs), file=fp)
        print('  from {:s};'.format(node), file=fp)
        print('  to {:s};'.format(xfmr_meter), file=fp)
        print('  configuration {:s};'.format(xfkey), file=fp)
        print('}', file=fp)
        if secondary_ft is not None:
            print('object {:s} {{'.format(meter_class), file=fp)
            print('  name {:s};'.format(xfmr_meter), file=fp)
            print('  phases {:s};'.format(phs), file=fp)
            print('  nominal_voltage {:.2f};'.format(vnom), file=fp)
            print('}', file=fp)
            print('object {:s} {{'.format(line_class), file=fp)
            print('  name {:s}_secondary;'.format(node), file=fp)
            print('  phases {:s};'.format(phs), file=fp)
            print('  from {:s};'.format(xfmr_meter), file=fp)
            print('  to {:s}_mtr;'.format(node), file=fp)
            print('  length {:.1f};'.format(secondary_ft), file=fp)
            print('  configuration {:s};'.format(linekey), file=fp)
            print('}', file=fp)

        print('object {:s} {{'.format(meter_class), file=fp)
        print('  name {:s}_mtr;'.format(node), file=fp)
        print('  phases {:s};'.format(phs), file=fp)
        print('  nominal_voltage {:.2f};'.format(vnom), file=fp)
        write_tariff(fp)
        if ConfigDict['metrics_interval']['value'] > 0:
            print('  object metrics_collector {', file=fp)
            print('    interval', str(ConfigDict['metrics_interval']['value']) + ';', file=fp)
            print('  };', file=fp)
        print('}', file=fp)
        # write all the houses on that meter
        ConfigDict['house_nodes']['value'][node] = [nhouse, region, lg_v_sm, phs, bldg, ti]
        write_houses(node, fp, vnom, bIgnoreThermostatSchedule=False, bWriteService=False, bTriplex=bTriplex,
                     setpoint_offset=1.0)
    else:
        print('// Zero houses at {:s} phases {:s}'.format(node, phs), file=fp)


# ***************************************************************************************************
# ***************************************************************************************************

def populate_feeder(configfile=None, config=None, taxconfig=None, fgconfig=None):
    """Wrapper function that processes one feeder. One or two keyword arguments must be supplied.

    Args:
        configfile (str): JSON file name for the feeder population data, mutually exclusive with config
        config (dict): dictionary of feeder population data already read in, mutually exclusive with configfile
        taxconfig (dict): dictionary of custom taxonomy data for ERCOT processing
        targetdir (str): directory to receive the output files, defaults to ./CaseName
    """
    global ConfigDict

    if configfile is not None:
        checkResidentialBuildingTable()
    # we want the same pseudo-random variables each time, for repeatability
    np.random.seed(0)

    if config is None:
        lp = open(configfile).read()
        config = json.loads(lp)
    if fgconfig is not None:
        fgfile = open(fgconfig).read()
        ConfigDict = json.loads(fgfile)

    rootname = config['BackboneFiles']['TaxonomyChoice']
    tespdir = os.path.expandvars(os.path.expanduser(config['SimulationConfig']['SourceDirectory']))
    ConfigDict['glmpath']['value'] = tespdir + '/feeders/'
    ConfigDict['supportpath']['value'] = ''  # tespdir + '/schedules'
    ConfigDict['climate']['weatherpath']['value'] = ''  # tespdir + '/weather'
    if 'NamePrefix' in config['BackboneFiles']:
        ConfigDict['name_prefix']['value'] = config['BackboneFiles']['NamePrefix']
    if 'WorkingDirectory' in config['SimulationConfig']:
        ConfigDict['outpath']['value'] = config['SimulationConfig']['WorkingDirectory'] + '/'  # for full-order DSOT
    #      outpath = './' + config['SimulationConfig']['CaseName'] + '/'
    else:
        # outpath = './' + config['SimulationConfig']['CaseName'] + '/'
        ConfigDict['outpath']['value'] = './' + config['SimulationConfig']['CaseName'] + '/'
#    ConfigDict['starttime']['value'] = config['SimulationConfig']['StartTime']
    ConfigDict['simtime']['starttime'] = config['SimulationConfig']['StartTime']
#    ConfigDict['endtime']['value'] = config['SimulationConfig']['EndTime']
    ConfigDict['simtime']['endtime'] = config['SimulationConfig']['EndTime']
    ConfigDict['timestep']['value'] = int(config['FeederGenerator']['MinimumStep'])
    ConfigDict['metrics_interval']['value'] = int(config['FeederGenerator']['MetricsInterval'])
    ConfigDict['electric_cooling_percentage']['value'] = 0.01 * float(
        config['FeederGenerator']['ElectricCoolingPercentage'])
    ConfigDict['water_heater_percentage']['value'] = 0.01 * float(config['FeederGenerator']['WaterHeaterPercentage'])
    ConfigDict['water_heater_participation']['value'] = 0.01 * float(
        config['FeederGenerator']['WaterHeaterParticipation'])
    ConfigDict['solar_percentage']['value'] = 0.01 * float(config['FeederGenerator']['SolarPercentage'])
    ConfigDict['storage_percentage']['value'] = 0.01 * float(config['FeederGenerator']['StoragePercentage'])
    ConfigDict['solar_inv_mode']['value'] = config['FeederGenerator']['SolarInverterMode']
    ConfigDict['storage_inv_mode']['value'] = config['FeederGenerator']['StorageInverterMode']
    ConfigDict['weather_file']['value'] = config['WeatherPrep']['DataSource']
    ConfigDict['billing']['bill_mode']['value'] = config['FeederGenerator']['BillingMode']
    ConfigDict['billing']['kwh_price']['value'] = float(config['FeederGenerator']['Price'])
    ConfigDict['billing']['monthly_fee']['value'] = float(config['FeederGenerator']['MonthlyFee'])
    ConfigDict['billing']['tier1_energy']['value'] = float(config['FeederGenerator']['Tier1Energy'])
    ConfigDict['billing']['tier1_price']['value'] = float(config['FeederGenerator']['Tier1Price'])
    ConfigDict['billing']['tier2_energy']['value'] = float(config['FeederGenerator']['Tier2Energy'])
    ConfigDict['billing']['tier2_price']['value'] = float(config['FeederGenerator']['Tier2Price'])
    ConfigDict['billing']['tier3_energy']['value'] = float(config['FeederGenerator']['Tier3Energy'])
    ConfigDict['billing']['tier3_price']['value'] = float(config['FeederGenerator']['Tier3Price'])
    ConfigDict['Eplus']['Eplus_Bus']['value'] = config['EplusConfiguration']['EnergyPlusBus']
    ConfigDict['Eplus']['Eplus_Volts']['value'] = float(config['EplusConfiguration']['EnergyPlusServiceV'])
    ConfigDict['Eplus']['Eplus_kVA']['value'] = float(config['EplusConfiguration']['EnergyPlusXfmrKva'])
    ConfigDict['xmfr']["transmissionXfmrMVAbase"]['value'] = float(config['PYPOWERConfiguration']['TransformerBase'])
    ConfigDict["transmissionVoltage"]['value'] = 1000.0 * float(config['PYPOWERConfiguration']['TransmissionVoltage'])
    ConfigDict['latitude']['value'] = float(config['WeatherPrep']['Latitude'])
    ConfigDict['longitude']['value'] = float(config['WeatherPrep']['Longitude'])
    ConfigDict['altitude']['value'] = float(config['WeatherPrep']['Altitude'])
    ConfigDict['tz_meridian']['value'] = float(config['WeatherPrep']['TZmeridian'])
    if 'AgentName' in config['WeatherPrep']:
        ConfigDict['climate']['weatherName']['value'] = config['WeatherPrep']['AgentName']

    ConfigDict['house_nodes']['value'] = {}
    ConfigDict['small_nodes']['value'] = {}
    ConfigDict['comm_loads']['value'] = {}

    if taxconfig is not None:
        print('called with a custom taxonomy configuration')
        forERCOT = True

        if rootname in taxconfig['backbone_feeders']:
            taxrow = taxconfig['backbone_feeders'][rootname]
            vll = taxrow['vll']
            vln = taxrow['vln']
            avg_house = taxrow['avg_house']
            avg_comm = taxrow['avg_comm']
            ConfigDict["fncs_case"]['value'] = config['SimulationConfig']['CaseName']
            ConfigDict['glmpath']['value'] = taxconfig['glmpath']
            ConfigDict['outpath']['value'] = taxconfig['outpath']
            ConfigDict['supportpath']['value'] = taxconfig['supportpath']
            ConfigDict['climate']['weatherpath']['value'] = taxconfig['weatherpath']
            print(ConfigDict['fncs_case']['value'], rootname, vll, vln, avg_house, avg_comm,
                  ConfigDict['glmpath']['value'], ConfigDict['outpath']['value'], ConfigDict['supportpath']['value'],
                  ConfigDict['climate']['weatherpath']['value'])
            ProcessTaxonomyFeeder(ConfigDict['fncs_case']['value'], rootname, vll, vln, avg_house,
                                  avg_comm)  # need a name_prefix mechanism
        else:
            print(rootname, 'not found in taxconfig backbone_feeders')
    else:
        print('using the built-in taxonomy')
        print(rootname, 'to', ConfigDict['outpath']['value'], 'using', ConfigDict['weather_file']['value'])
#        print('times', ConfigDict['starttime']['value'], ConfigDict['endtime']['value'])
        print('times', ConfigDict['simtime']['starttime'], ConfigDict['simtime']['endtime'])
        print('steps', ConfigDict['timestep']['value'], ConfigDict['metrics_interval']['value'])
        print('hvac', ConfigDict['electric_cooling_percentage']['value'])
        print('pv', ConfigDict['solar_percentage']['value'], ConfigDict['solar_inv_mode']['value'])
        print('storage', ConfigDict['storage_percentage']['value'], ConfigDict['storage_inv_mode']['value'])
        print('billing', ConfigDict['billing']['kwh_price']['value'], ConfigDict['billing']['monthly_fee']['value'])
        for c in ConfigDict['taxchoice']['value']:
            if c[0] == rootname:
                ConfigDict['fncs_case']['value'] = config['SimulationConfig']['CaseName']
                ProcessTaxonomyFeeder(ConfigDict['fncs_case']['value'], c[0], c[1], c[2], c[3], c[4])


#                quit()

# ***************************************************************************************************
# ***************************************************************************************************

def populate_all_feeders():
    """Wrapper function that batch processes all taxonomy feeders in the casefiles table (see source file)
    """
    print(ConfigDict['casefiles'])
    
    #if sys.platform == 'win32':
    #    batname = 'run_all.bat'
    #else:
    #    batname = 'run_all.sh'
    # op = open(ConfigDict['outpath']['value'] + batname, 'w')
    # for c in ConfigDict['casefiles']:
    #     print('gridlabd -D WANT_VI_DUMP=1 -D METRICS_FILE=' + c[0] + '.json', c[0] + '.glm', file=op)
    # op.close()
    outname = ConfigDict['casefiles']['outname']
    
    ProcessTaxonomyFeeder(outname,
    				ConfigDict['casefiles']['rootname'],
    				ConfigDict['casefiles']['vll'],
    				ConfigDict['casefiles']['vln'],
    				ConfigDict['casefiles']['avghouse'],
    				ConfigDict['casefiles']['avgcommercial'])
        
        #def ProcessTaxonomyFeeder(outname, rootname, vll, vln, avghouse, avgcommercial):



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='populates GLD model with houses')
	parser.add_argument('-c', '--config_file',
                help='JSON config file defining how the model is to be populated',
                nargs='?',
                default = './FeederGenerator.json')
	args = parser.parse_args()
	initialize_config_dict(args.config_file)
	populate_all_feeders()
	
