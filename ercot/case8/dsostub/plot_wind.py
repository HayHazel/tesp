#   Copyright (C) 2017-2018 Battelle Memorial Institute
# file: process_pypower.py
import json;
#import sys;
import numpy as np;
import matplotlib as mpl;
import matplotlib.pyplot as plt;

def bus_color(key):
    if key == '1':
        return 'b'
    if key == '2':
        return 'g'
    if key == '3':
        return 'r'
    if key == '4':
        return 'c'
    if key == '5':
        return 'm'
    if key == '6':
        return 'y'
    if key == '7':
        return 'k'
    if key == '8':
        return 'cadetblue'
    return 'k'

def unit_width(dict, key):
    if dict['generators'][key]['bustype'] == 'swing':
        return 2.0
    return 1.0

def unit_color(dict, key):
    genfuel = dict['generators'][key]['genfuel']
    if genfuel == 'wind':
        return 'g'
    if genfuel == 'nuclear':
        return 'r'
    if genfuel == 'coal':
        return 'k'
    if genfuel == 'gas':
        return 'b'
    return 'y'

def process_pypower(nameroot):
    # first, read and print a dictionary of relevant PYPOWER objects
    lp = open (nameroot + '_m_dict.json').read()
    dict = json.loads(lp)
    baseMVA = dict['baseMVA']
    gen_keys = list(dict['generators'].keys())
    gen_keys.sort()
    print('\nGenerator Dictionary:')
    print('Unit Bus Type Fuel Pmin Pmax Costs[Start Stop C2 C1 C0]')
    for key in gen_keys:
        row = dict['generators'][key]
        print (key, row['bus'], row['bustype'], row['genfuel'], row['Pmin'], row['Pmax'], '[', row['StartupCost'], row['ShutdownCost'], row['c2'], row['c1'], row['c0'], ']')

    # read the generator metrics file
    lp_g = open ('gen_' + nameroot + '_metrics.json').read()
    lst_g = json.loads(lp_g)
    print ('\nGenerator Metrics data starting', lst_g['StartTime'])
    # make a sorted list of the times, and NumPy array of times in hours
    lst_g.pop('StartTime')
    meta_g = lst_g.pop('Metadata')
    times = list(map(int,list(lst_g.keys())))
    times.sort()
    hrs = np.array(times, dtype=np.float)
    denom = 3600.0
    hrs /= denom

    print ('\nGenerator Metadata [Variable Index Units] for', len(lst_g[str(times[0])]), 'objects')
    for key, val in meta_g.items():
        print (key, val['index'], val['units'])
        if key == 'Pgen':
            PGEN_IDX = val['index']
            PGEN_UNITS = val['units']
        elif key == 'Qgen':
            QGEN_IDX = val['index']
            QGEN_UNITS = val['units']
        elif key == 'LMP_P':
            GENLMP_IDX = val['index']
            GENLMP_UNITS = val['units']

    # create a NumPy array of all generator metrics
    data_g = np.empty(shape=(len(gen_keys), len(times), len(lst_g[str(times[0])][gen_keys[0]])), dtype=np.float)
    print ('\nConstructed', data_g.shape, 'NumPy array for Generators')
    print ('Unit,Bus,Type,Fuel,CF,COV')
    j = 0
    for key in gen_keys:
        i = 0
        for t in times:
            ary = lst_g[str(t)][gen_keys[j]]
            data_g[j, i,:] = ary
            i = i + 1
        p_avg = data_g[j,:,PGEN_IDX].mean()
        p_std = data_g[j,:,PGEN_IDX].std()
        row = dict['generators'][key]
        p_max = float (row['Pmax'])
        CF = p_avg/p_max
        if p_avg > 0.0:
            COV = p_std/p_avg
        else:
            COV = 0.0

        print (key, row['bus'], row['bustype'], row['genfuel'],
               '{:.4f}'.format (CF),
               '{:.4f}'.format (COV))
        j = j + 1

    # display a plot 
    fig, ax = plt.subplots(1, 1, sharex = 'col')
    tmin = 0.0
    tmax = 48.0
    xticks = [0,6,12,18,24,30,36,42,48]
    ax.grid (linestyle = '-')
    ax.set_xlim(tmin,tmax)
    ax.set_xticks(xticks)
    ax.set_xlabel('Hours')

    ax.set_title ('Wind Generator Outputs')
    ax.set_ylabel(PGEN_UNITS)
    for i in range(data_g.shape[0]):
        genfuel = dict['generators'][gen_keys[i]]['genfuel']
        bus = dict['generators'][gen_keys[i]]['bus']
        print (genfuel, bus)
        if genfuel == 'wind':
            ax.plot(hrs, data_g[i,:,PGEN_IDX], color=bus_color (str(bus)))

    plt.show()

if __name__ == '__main__':
    process_pypower ('ercot_8')