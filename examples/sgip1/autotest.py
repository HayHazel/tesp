# Copyright (C) 2017-2020 Battelle Memorial Institute
# file: autotest.py
"""Runs the set of TESP test cases
"""
import sys
import subprocess
import os
import stat
import shutil

if sys.platform == 'win32':
    pycall = 'python'
else:
    pycall = 'python3'

def ProcessLine(line, local_vars):
#  print ('@@@@ input line to execute:', line)
  foreground = line.replace (' &)', ')').replace(' &>', ' >')
  exports = ''
  for var in local_vars:
    exports = exports + 'export ' + var['key'] + '=' + var['val'] + ' && '
#  print (' line transformed to:', exports + foreground)
  return exports + foreground

"""Runs a test case based on pre-existing shell script file.

Waits for the FNCS or HELICS broker process to finish before function returns.

"""
def RunTestCase(fname):
  local_vars = []
  fp = open (fname, 'r')
  potherList=[]
  # if a HELICS case includes EnergyPlus, both brokers will instantiate
  pFNCSbroker = None
  pHELICSbroker = None
  for ln in fp:
    line = ln.rstrip('\n')
    if ('#!/bin/bash' in line) or (len(line) < 1):
      continue
    if line[0] == '#':
      continue
    if line.startswith('declare'):
      toks = line.split()
      keyval = toks[2].split('=')
      local_vars.append({'key':keyval[0],'val':keyval[1]})
    elif line.startswith('javac') or line.startswith('python') or line.startswith('gridlabd') or line.startswith('TMY3toTMY2_ansi'):
      jc = subprocess.Popen (ProcessLine (line, local_vars), shell=True)
      jc.wait()
    elif 'fncs_broker' in line:
      pFNCSbroker = subprocess.Popen (ProcessLine (line, local_vars), shell=True)
    elif 'helics_broker' in line:
      pHELICSbroker = subprocess.Popen (ProcessLine (line, local_vars), shell=True)
    else:
      pother = subprocess.Popen (ProcessLine (line, local_vars), shell=True)
      potherList.append(pother)
  fp.close()
  if pFNCSbroker is not None:
    pFNCSbroker.wait()
    print ('====   Fncs Broker Exit in', os.getcwd())
  if pHELICSbroker is not None:
    pHELICSbroker.wait()
    print ('==== Helics Broker Exit in', os.getcwd())
  for p in potherList:
    p.wait()
  print   ('================== Exit in', os.getcwd())

if __name__ == '__main__':
  print('start examples sgip1: ')
  p1 = subprocess.Popen ('./clean.sh', shell=True)
  p1.wait()
  p1 = subprocess.Popen (pycall + ' prepare_cases.py', shell=True)
  p1.wait()
  RunTestCase ('runSGIP1a.sh')
  RunTestCase ('runSGIP1b.sh')
  RunTestCase ('runSGIP1c.sh')
  RunTestCase ('runSGIP1d.sh')
  RunTestCase ('runSGIP1e.sh')
  RunTestCase ('runSGIP1ex.sh')

