# Copyright (C) 2021-2022 Battelle Memorial Institute
# file: loopERCOT.py

import json
import math
from copy import deepcopy
import numpy as np
import pypower.api as pp
import scipy.interpolate as ip

import tesp_support.tso_helpers as tso

casename = 'ercot_8'
wind_period = 3600

load_shape = [0.6704,
              0.6303,
              0.6041,
              0.5902,
              0.5912,
              0.6094,
              0.6400,
              0.6725,
              0.7207,
              0.7584,
              0.7905,
              0.8171,
              0.8428,
              0.8725,
              0.9098,
              0.9480,
              0.9831,
              1.0000,
              0.9868,
              0.9508,
              0.9306,
              0.8999,
              0.8362,
              0.7695,
              0.6704]  # wrap to the next day


def rescale_case(ppc, scale):
    ppc['bus'][:, 2] *= scale  # Pd
    ppc['bus'][:, 3] *= scale  # Qd
    ppc['bus'][:, 5] *= (scale * scale)  # Qs
    ppc['gen'][:, 1] *= scale  # Pg
    return


# from 'ARIMA-Based Time Series Model of Stochastic Wind Power Generation'
# return dict with rows like wind['unit'] = [bus, MW, Theta0, Theta1, StdDev, Psi1, Ylim, alag, ylag, p]
def make_wind_plants(ppc):
    plants = {}
    Pnorm = 165.6
    gen = ppc['gen']
    cost = ppc['gencost']
    for i in range(gen.shape[0]):
        busnum = int(gen[i, 0])
        c2 = float(cost[i, 4])
        if c2 < 2e-5:  # genfuel would be 'wind'
            MW = float(gen[i, 8])
            scale = MW / Pnorm
            Theta0 = 0.05 * math.sqrt(scale)
            Theta1 = -0.1 * (scale)
            StdDev = math.sqrt(1.172 * math.sqrt(scale))
            Psi1 = 1.0
            Ylim = math.sqrt(MW)
            alag = Theta0
            ylag = Ylim
            plants[str(i)] = [busnum, MW, Theta0, Theta1, StdDev, Psi1, Ylim, alag, ylag, MW]
    return plants


# this differs from tesp_support because of additions to FNCS, and Pnom==>Pmin for generators
def make_dictionary(ppc, rootname):
    fncsBuses = {}
    generators = {}
    unitsout = []
    branchesout = []
    bus = ppc['bus']
    gen = ppc['gen']
    cost = ppc['gencost']
    fncsBus = ppc['DSO']
    units = ppc['UnitsOut']
    branches = ppc['BranchesOut']

    for i in range(gen.shape[0]):
        busnum = int(gen[i, 0])
        bustype = bus[busnum - 1, 1]
        if bustype == 1:
            bustypename = 'pq'
        elif bustype == 2:
            bustypename = 'pv'
        elif bustype == 3:
            bustypename = 'swing'
        else:
            bustypename = 'unknown'
        gentype = 'other'  # as opposed to simple cycle or combined cycle
        c2 = float(cost[i, 4])
        c1 = float(cost[i, 5])
        c0 = float(cost[i, 6])
        if c2 < 2e-5:  # assign fuel types from the IA State default costs
            genfuel = 'wind'
        elif c2 < 0.0003:
            genfuel = 'nuclear'
        elif c1 < 25.0:
            genfuel = 'coal'
        else:
            genfuel = 'gas'
        generators[str(i + 1)] = {'bus': int(busnum), 'bustype': bustypename, 'Pmin': float(gen[i, 9]),
                                  'Pmax': float(gen[i, 8]), 'genfuel': genfuel, 'gentype': gentype,
                                  'StartupCost': float(cost[i, 1]), 'ShutdownCost': float(cost[i, 2]),
                                  'c2': c2, 'c1': c1, 'c0': c0}

    for i in range(fncsBus.shape[0]):
        busnum = int(fncsBus[i, 0])
        busidx = busnum - 1
        fncsBuses[str(busnum)] = {'Pnom': float(bus[busidx, 2]), 'Qnom': float(bus[busidx, 3]),
                                  'area': int(bus[busidx, 6]), 'zone': int(bus[busidx, 10]),
                                  'ampFactor': float(fncsBus[i, 2]), 'GLDsubstations': [fncsBus[i, 1]],
                                  'curveScale': float(fncsBus[i, 5]), 'curveSkew': int(fncsBus[i, 6])}

    for i in range(units.shape[0]):
        unitsout.append({'unit': int(units[i, 0]), 'tout': int(units[i, 1]), 'tin': int(units[i, 2])})

    for i in range(branches.shape[0]):
        branchesout.append({'branch': int(branches[i, 0]), 'tout': int(branches[i, 1]), 'tin': int(branches[i, 2])})

    dp = open(rootname + '_m_dict.json', 'w')
    ppdict = {'baseMVA': ppc['baseMVA'], 'fncsBuses': fncsBuses, 'generators': generators, 'UnitsOut': unitsout,
              'BranchesOut': branchesout}
    print(json.dumps(ppdict), file=dp, flush=True)
    dp.close()


x = np.array(range(25))
y = np.array(load_shape)
l = len(x)
t = np.linspace(0, 1, l - 2, endpoint=True)
t = np.append([0, 0, 0], t)
t = np.append(t, [1, 1, 1])
tck_load = [t, [x, y], 3]
u3 = np.linspace(0, 1, num=int(86400 / 300) + 1, endpoint=True)
newpts = ip.splev(u3, tck_load)

ppc = tso.load_json_case(casename + '.json')
ppopt_market = pp.ppoption(VERBOSE=0, OUT_ALL=0, PF_DC=ppc['pf_dc'])
ppopt_regular = pp.ppoption(VERBOSE=0, OUT_ALL=0, PF_DC=ppc['pf_dc'], PF_MAX_IT=20, PF_ALG=1)
StartTime = ppc['StartTime']
tmax = int(ppc['Tmax'])
period = int(ppc['Period'])
dt = int(ppc['dt'])
swing_bus = int(ppc['swing_bus'])

period = 900
dt = 180

# initialize for metrics collection
bus_mp = open('bus_' + casename + '_metrics.json', 'w')
gen_mp = open('gen_' + casename + '_metrics.json', 'w')
sys_mp = open('sys_' + casename + '_metrics.json', 'w')
bus_meta = {'LMP_P': {'units': 'USD/kwh', 'index': 0},
            'LMP_Q': {'units': 'USD/kvarh', 'index': 1},
            'PD': {'units': 'MW', 'index': 2},
            'QD': {'units': 'MVAR', 'index': 3},
            'Vang': {'units': 'deg', 'index': 4},
            'Vmag': {'units': 'pu', 'index': 5},
            'Vmax': {'units': 'pu', 'index': 6},
            'Vmin': {'units': 'pu', 'index': 7}}
gen_meta = {'Pgen': {'units': 'MW', 'index': 0},
            'Qgen': {'units': 'MVAR', 'index': 1},
            'LMP_P': {'units': 'USD/kwh', 'index': 2}}
sys_meta = {'Ploss': {'units': 'MW', 'index': 0},
            'Converged': {'units': 'true/false', 'index': 1}}
bus_metrics = {'Metadata': bus_meta, 'StartTime': StartTime}
gen_metrics = {'Metadata': gen_meta, 'StartTime': StartTime}
sys_metrics = {'Metadata': sys_meta, 'StartTime': StartTime}
make_dictionary(ppc, casename)
tnext_metrics = 0
loss_accum = 0
conv_accum = True
n_accum = 0
bus_accum = {}
gen_accum = {}
fncsBus = ppc['DSO']
gen = ppc['gen']
for i in range(fncsBus.shape[0]):
    busnum = int(fncsBus[i, 0])
    bus_accum[str(busnum)] = [0, 0, 0, 0, 0, 0, 0, 99999.0]
for i in range(gen.shape[0]):
    gen_accum[str(i + 1)] = [0, 0, 0]

# ppc arrays (bus type 1=load, 2 = gen (PV) and 3 = swing)
# bus: bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin
# gen: bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, (11 zeros)
# branch: fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
# gencost: 2, startup, shutdown, 3, c2, c1, c0
# UnitsOut: idx, time out[s], time back in[s]
# BranchesOut: idx, time out[s], time back in[s]
# FNCS: bus, topic, gld_scale, Pnom, Qnom, curve_scale, curve_skew
fncs_bus = ppc['DSO']
loads = {'h': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': []}

# initialize for variable wind
tnext_wind = tmax + 2 * dt  # by default, never fluctuate the wind plants
if wind_period > 0:
    wind_plants = make_wind_plants(ppc)
    if len(wind_plants) < 1:
        print('warning: wind power fluctuation requested, but there are no wind plants in this case')
    else:
        tnext_wind = 0
print(wind_plants)
# initialize for OPF and time stepping
ts = 0
tnext_opf = 0

op = open(casename + '.csv', 'w')
print('seconds,OPFconverged,TotalLoad,TotalGen,SwingGen,LMP1,LMP8,gas1,coal1,nuc1,gas2,' +
      'coal2,nuc2,gas3,coal3,gas4,gas5,coal5,gas7,coal7,wind1,wind3,wind4,wind6,wind7',
      sep=',', file=op, flush=True)
while ts <= tmax:
    # fluctuate the wind plants
    if ts >= tnext_wind:
        wind_MW = 0.0
        for key, row in wind_plants.items():
            # return dict with rows like
            # wind['unit'] = [bus, MW, Theta0, Theta1, StdDev, Psi1, Ylim, alag, ylag, p]
            wind_bus = row[0]
            MW = row[1]
            Theta0 = row[2]
            Theta1 = row[3]
            StdDev = row[4]
            Psi1 = row[5]
            Ylim = row[6]
            alag = row[7]
            ylag = row[8]
            p = row[9]
            if ts > 0:
                a = np.random.normal(0.0, StdDev)
                y = Theta0 + a - Theta1 * alag + Psi1 * ylag
                alag = a
            else:
                y = ylag
            if y > Ylim:
                y = Ylim
            elif y < 0.0:
                y = 0.0
            p = y * y
            if ts > 0:
                ylag = y
            row[7] = alag
            row[8] = ylag
            row[9] = p
            wind_MW += p
            # reset the unit capacity; this will 'stick' for the next wind_period
            ppc['gen'][int(key), 8] = p
            if ppc['gen'][int(key), 1] > p:
                ppc['gen'][int(key), 1] = p
        tnext_wind += wind_period
        print('{:6d} # {:d} wind plants produce {:.2f} MW'.format(ts, len(wind_plants), wind_MW))

    # always update the unresponsive load
    #  loads['h'].append (float(ts) / 3600.0)
    for row in fncs_bus:
        bus = int(row[0])
        topic = str(row[1])
        gld_scale = float(row[2])
        Pnom = float(row[3])
        Qnom = float(row[4])
        curve_scale = float(row[5])
        curve_skew = int(row[6])

        sec = (ts + curve_skew) % 86400
        h = float(sec) / 3600.0
        val = ip.splev([h / 24.0], tck_load)
        Pload = Pnom * val[1]
        Qload = Qnom * val[1]
        #    loads[str(bus)].append(Pload)
        ppc['bus'][bus - 1, 2] = Pload
        ppc['bus'][bus - 1, 3] = Qload
    # run OPF to establish the prices and economic dispatch
    if ts >= tnext_opf:
        ropf = pp.runopf(ppc, ppopt_market)
        if ropf['success'] == False:
            conv_accum = False
        opf_bus = deepcopy(ropf['bus'])
        opf_gen = deepcopy(ropf['gen'])
        Pswing = 0
        for idx in range(opf_gen.shape[0]):
            if opf_gen[idx, 0] == swing_bus:
                Pswing += opf_gen[idx, 1]
        print(ts, ropf['success'],
              '{:.2f}'.format(opf_bus[:, 2].sum()),
              '{:.2f}'.format(opf_gen[:, 1].sum()),
              '{:.2f}'.format(Pswing),
              '{:.4f}'.format(opf_bus[0, 13]),
              '{:.4f}'.format(opf_bus[7, 13]),
              '{:.2f}'.format(opf_gen[0, 1]),
              '{:.2f}'.format(opf_gen[1, 1]),
              '{:.2f}'.format(opf_gen[2, 1]),
              '{:.2f}'.format(opf_gen[3, 1]),
              '{:.2f}'.format(opf_gen[4, 1]),
              '{:.2f}'.format(opf_gen[5, 1]),
              '{:.2f}'.format(opf_gen[6, 1]),
              '{:.2f}'.format(opf_gen[7, 1]),
              '{:.2f}'.format(opf_gen[8, 1]),
              '{:.2f}'.format(opf_gen[9, 1]),
              '{:.2f}'.format(opf_gen[10, 1]),
              '{:.2f}'.format(opf_gen[11, 1]),
              '{:.2f}'.format(opf_gen[12, 1]),
              '{:.2f}'.format(opf_gen[13, 1]),
              '{:.2f}'.format(opf_gen[14, 1]),
              '{:.2f}'.format(opf_gen[15, 1]),
              '{:.2f}'.format(opf_gen[16, 1]),
              '{:.2f}'.format(opf_gen[17, 1]),
              sep=',', file=op, flush=True)
        tnext_opf += period

    # always run the regular power flow for voltages and performance metrics
    ppc['bus'][:, 13] = opf_bus[:, 13]  # set the lmp
    ppc['gen'][:, 1] = opf_gen[:, 1]  # set the economic dispatch
    rpf = pp.runpf(ppc, ppopt_regular)
    if rpf[0]['success'] == False:
        conv_accum = False
        print('rpf did not converge at', ts)
    #   pp.printpf (100.0,
    #               bus=rpf[0]['bus'],
    #               gen=rpf[0]['gen'],
    #               branch=rpf[0]['branch'],
    #               fd=sys.stdout,
    #               et=rpf[0]['et'],
    #               success=rpf[0]['success'])
    bus = rpf[0]['bus']
    gen = rpf[0]['gen']
    fncsBus = ppc['DSO']
    Pload = bus[:, 2].sum()
    Pgen = gen[:, 1].sum()
    Ploss = Pgen - Pload

    # update the metrics
    n_accum += 1
    loss_accum += Ploss
    for i in range(fncsBus.shape[0]):
        busnum = int(fncsBus[i, 0])
        busidx = busnum - 1
        row = bus[busidx].tolist()
        # LMP_P, LMP_Q, PD, QD, Vang, Vmag, Vmax, Vmin: row[11] and row[12] are Vmax and Vmin constraints
        PD = row[2]  # + resp # TODO, if more than one FNCS bus, track scaled_resp separately
        Vpu = row[7]
        bus_accum[str(busnum)][0] += row[13] * 0.001
        bus_accum[str(busnum)][1] += row[14] * 0.001
        bus_accum[str(busnum)][2] += PD
        bus_accum[str(busnum)][3] += row[3]
        bus_accum[str(busnum)][4] += row[8]
        bus_accum[str(busnum)][5] += Vpu
        if Vpu > bus_accum[str(busnum)][6]:
            bus_accum[str(busnum)][6] = Vpu
        if Vpu < bus_accum[str(busnum)][7]:
            bus_accum[str(busnum)][7] = Vpu
    for i in range(gen.shape[0]):
        row = gen[i].tolist()
        busidx = int(row[0] - 1)
        # Pgen, Qgen, LMP_P  (includes the responsive load as dispatched by OPF)
        gen_accum[str(i + 1)][0] += row[1]
        gen_accum[str(i + 1)][1] += row[2]
        gen_accum[str(i + 1)][2] += float(opf_bus[busidx, 13]) * 0.001

    # write the metrics
    if ts >= tnext_metrics:
        sys_metrics[str(ts)] = {casename: [loss_accum / n_accum, conv_accum]}

        bus_metrics[str(ts)] = {}
        for i in range(fncsBus.shape[0]):
            busnum = int(fncsBus[i, 0])
            busidx = busnum - 1
            row = bus[busidx].tolist()
            met = bus_accum[str(busnum)]
            bus_metrics[str(ts)][str(busnum)] = [met[0] / n_accum, met[1] / n_accum, met[2] / n_accum, met[3] / n_accum,
                                                 met[4] / n_accum, met[5] / n_accum, met[6], met[7]]
            bus_accum[str(busnum)] = [0, 0, 0, 0, 0, 0, 0, 99999.0]

        gen_metrics[str(ts)] = {}
        for i in range(gen.shape[0]):
            met = gen_accum[str(i + 1)]
            gen_metrics[str(ts)][str(i + 1)] = [met[0] / n_accum, met[1] / n_accum, met[2] / n_accum]
            gen_accum[str(i + 1)] = [0, 0, 0]

        tnext_metrics += period
        n_accum = 0
        loss_accum = 0
        conv_accum = True

    ts += dt

# ======================================================
print('writing metrics', flush=True)
print(json.dumps(sys_metrics), file=sys_mp, flush=True)
print(json.dumps(bus_metrics), file=bus_mp, flush=True)
print(json.dumps(gen_metrics), file=gen_mp, flush=True)
print('closing files', flush=True)
bus_mp.close()
gen_mp.close()
sys_mp.close()
op.close()

# fig, ax = plt.subplots()
# ax.set_title ('Base Non-responsive Load Shape: Smoothed Peak = ' + '{:.4f}'.format (max(newpts[1])))
# ax.set_ylabel ('Per-unit Power')
# ax.set_xlabel ('Hours')
# ax.grid (linestyle = '-')
# ax.plot (x, y, 'bo-')
# ax.plot (newpts[0], newpts[1], 'r')
# plt.show()

# fig, ax = plt.subplots()
# tmin = 0.0
# tmax = 48.0
# xticks = [0,6,12,18,24,30,36,42,48]
# ax.set_title ('Non-responsive Loads')
# ax.plot (loads['h'], loads['1'], 'b', label='bus1')
# ax.plot (loads['h'], loads['2'], 'g', label='bus2')
# ax.plot (loads['h'], loads['3'], 'r', label='bus3')
# ax.plot (loads['h'], loads['4'], 'c', label='bus4')
# ax.plot (loads['h'], loads['5'], 'm', label='bus5')
# ax.plot (loads['h'], loads['6'], 'y', label='bus6')
# ax.plot (loads['h'], loads['7'], 'k', label='bus7')
# ax.plot (loads['h'], loads['8'], 'cadetblue', label='bus8')
# ax.legend ()
# ax.set_ylabel ('Real Power [MW]')
# ax.set_xlabel ('Hours')
# ax.grid (linestyle = '-')
# ax.set_xlim(tmin,tmax)
# ax.set_xticks(xticks)
# plt.show()
#
