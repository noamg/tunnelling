# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:55:09 2016

@author: noam
"""

import os

import numpy as np
import scipy as sp
from scipy import io

import matplotlib.pyplot as plt

f_I_V_H = '/home/noam/studies/physics/amirim_hadar/tunnelling.git/measurements/JCUL-0103.mat'
f_I_V = '/home/noam/studies/physics/amirim_hadar/tunnelling.git/measurements/JCUL-0009.mat'
f_I_H_keithley = '/home/noam/studies/physics/amirim_hadar/tunnelling.git/measurements/JCUL-0017.mat'
f_I_H_magnet = '/home/noam/studies/physics/amirim_hadar/tunnelling.git/measurements/JCUL-0057.mat'
f_I_record = ''

figs_dir = '/home/noam/studies/physics/amirim_hadar/tunnelling.git/figs/'
data_dir = '/home/noam/studies/physics/amirim_hadar/tunnelling.git/measurements/'
data_prefix = 'JCUL-'
data_format = '.mat'
fig_format = '.svg'
print(os.path.join(figs_dir + '0000' + fig_format))
print(os.path.join(data_dir + 'JCUL-' + '0103' + '.mat'))
assert os.path.join(data_dir + data_prefix + '0103' + data_format) == '/home/noam/studies/physics/amirim_hadar/tunnelling.git/measurements/JCUL-0103.mat'


N_I_V_H = '0103'
N_I_V = '0009'
N_I_H_keithley = '0017'
N_I_H_magnet = '0057'
N_I_record = ''

def read_mesdata(f):
    out = sp.io.loadmat(f)
    M = out['mesdata']
    val = M[0,0]
    data = val['data']
    sweeped = list(map(lambda a: a[0], val['sweeped'].flatten()))
    measured = list(map(lambda a: a[0], val['measured'].flatten()))
    measured_time = val['measurement_time'][0]

    assert data.shape != (0, 0), 'data is empty'
    assert data.shape[1] == len(sweeped) + len(measured), 'number of fields stated unlike data columns'
    
    if 'lakes336.ctemp' in measured:
        T = data[:, len(sweeped) + measured.index('lakes336.ctemp')]
    else:
        T = np.array([-10, -10])
    return sweeped, measured, data, measured_time, [T.min(), T.max()]

def read_I_V(f, amp, Vdc_div):
    sweeped, measured, data, measured_time, T_range = read_mesdata(f)
    assert 'duck.AC0DC' in sweeped
    Vdc = data[:, sweeped.index('duck.AC0DC')] / Vdc_div
    if 'dmm1.dcv' in measured[1]:
        Idc = data[:, len(sweeped) + 1] / amp
        
    return Vdc, Idc
    
def read_I_H(f, amp, tesla_ampere_ratio=1207.7e-4):
    sweeped, measured, data, measured_time, T_range = read_mesdata(f)
    if sweeped[0] == 'keithley1.dcc':
        H = data[:, 0] * tesla_ampere_ratio
    elif sweeped[0] == 'time':
        H = data[:, len(sweeped) + measured.index('magnet.field')]
    if 'dmm1.dcv' in measured[1]:
        Idc = data[:, len(sweeped) + 1] / amp
    return H, Idc
        
        
    return Vdc, Idc
    pass

    
    

def read_record(f, amp):
    sweeped, measured, data, measured_time, T_range = read_mesdata(f)
    assert sweeped[0] == 'time'
    time = data[:, 0]
    if 'dmm1.dcv' in measured[1]:
        Idc = data[:, len(sweeped) + 1] / amp
        
    return time, Idc
    
#%%
sweeped, measured, data, measured_time, T_range = read_mesdata(f_I_H_magnet)
#def process_I_H(....I_min, dV)

def test_read_mesdata():
    sweeped, measured, data, measured_time, T_range = read_mesdata(f_I_V_H)
    print(measured)
    print(sweeped)
    print(data.shape)
    print(T_range)

def test_read_I_V():
    Vdc, Idc = read_I_V(f_I_V, amp=1e9, Vdc_div=100)
    _ = plt.figure()
    _ = plt.plot(Vdc, Idc, '.')

def test_read_I_H():
    H, Idc = read_I_H(f_I_H_keithley, amp=1e9)
    _ = plt.figure()
    _ = plt.plot(H, Idc, '.')

    H, Idc = read_I_H(f_I_H_magnet, amp=1e9)
    _ = plt.figure()
    _ = plt.plot(H, Idc, '.')

if False:
    test_read_mesdata()
    test_read_I_V()
    test_read_I_H()

def process_I_V(f, amp, Vdc_div=100, SF=11, axes=None, label=''):
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    mask = np.ones(SF) * 1.0 / SF
    Vdc, Idc = read_I_V(f, amp, Vdc_div)
    dIdVdc = np.diff(Idc) / np.diff(Vdc)
    dIdVdc = np.convolve(dIdVdc, mask, mode='same')

    N = np.int((SF - 1) * 0.5)
    dIdVdc[0:N] = dIdVdc[0:N] * np.arange(SF-1, N, -1) / N
    dIdVdc[-N:] = dIdVdc[-N:] * np.arange(N, SF-1) / N
    
    ax1, ax2 = axes
    _ = ax1.plot(Vdc * 1e3, Idc * 1e9, '.', label=label)
    ax1.set_xlabel('V[mV]')
    ax1.set_ylabel('I[nA]')
    _ = ax2.plot(Vdc[1:,] * 1e3, dIdVdc * 1e6, '.', label=label)
    ax2.set_xlabel('V[mV]')
    ax2.set_ylabel(r'G[$\mu$S]')

def process_I_H(f, amp, I_min, dV, ax=None, label=''):
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    H, Idc = read_I_H(f, amp)
    dIdV_zero_bias = (Idc - I_min) / dV
    _ = ax.plot(H, dIdV_zero_bias * 1e9, '.', label=label)
    ax.set_xlabel('H[T]')
    ax.set_ylabel('G[nS]')

def process_record(f, amp, I_min, dV, ax=None, label=''):
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    time, Idc = read_record(f, amp)
    dIdV_zero_bias = (Idc - I_min) / dV
    #_ = ax.plot(time, dIdV_zero_bias * 1e9, '.', label=label)
    _ = ax.plot(time[::2], dIdV_zero_bias[::2] * 1e9, '.', label=label+'even')
    _ = ax.plot(time[1::2], dIdV_zero_bias[1::2] * 1e9, '.', label=label+'odd')
    ax.set_xlabel('time[s]')
    ax.set_ylabel('G[nS]')

#%%    
     
if False:
    fig, axes = plt.subplots(2, 1, sharex=True)     
    process_I_V(os.path.join(data_dir + data_prefix + N_I_V + data_format), amp=1e9, axes=axes, label='label')
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    fig.savefig(os.path.join(figs_dir + N_I_V + fig_format))
    
    fig = plt.figure()
    ax = plt.subplot(111)
    process_I_H(os.path.join(data_dir + data_prefix + N_I_H_keithley + data_format), amp=1e9, I_min=0, dV=100e-6, ax=ax, label='label')
    ax.legend(loc='best')
    fig = plt.figure()
    ax = plt.subplot(111)
    process_I_H(os.path.join(data_dir + data_prefix + N_I_H_magnet + data_format), 1e9, 0, 100e-6, ax, label='label')
    ax.legend(loc='best')

#%% dIdV low bias few T
fig, axes = plt.subplots(2, 1, sharex=True)
Ns = ['0001', '0002', '0005', '0010']
amps = [1e9, 1e9, 1e9, 1e9]
labels = ['4K', '4K', '2.2K', '0.3K']
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    process_I_V(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, axes=axes, label=label)
axes[0].legend(loc='best')
axes[1].legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'conductance_low_bias_few_T' + fig_format))



#%% didV low bias few T, more.
# maybe should shift the minimum to v = 0
fig, axes = plt.subplots(2, 1, sharex=True)
Ns = ['0001', '0002', '0005', '0010', '0012', '0013', '0016']
amps = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e10]
labels = ['4K', '4K', '2.2K', '0.3K', '0.3K', '0.3K', '0.3K']
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    process_I_V(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, axes=axes, label=label)
axes[0].legend(loc='best')
axes[1].legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'conductance_low_bias_few_T_12_13' + fig_format))




#%% dIdV high bias few T
# shoft V = 0
fig, axes = plt.subplots(2, 1, sharex=True)
Ns = ['0003', '0004', '0006', '0014']
amps = [1e8, 1e8, 1e9, 1e8]
labels = ['4K', '4K', '1.7K', '0.3K']
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    if N == '0006':
        process_I_V(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, axes=axes, label=label, SF=43)
    else:
        process_I_V(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, axes=axes, label=label)
axes[0].legend(loc='best')
axes[1].legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'conductance_high_bias_few_T' + fig_format))

#%% ZBC low H, should fix I_min, dV

fig = plt.figure()
ax = plt.subplot(111)
Ns = ['0017', '0019']
amps = [1e10, 1e10,]
labels = ['up', 'down']
I_mins = [0, 0,]
dVs = np.array([100e-6, 100e-6])
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    I_min = I_mins[i]
    dV = dVs[i]
    process_I_H(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, I_min=I_min, dV=dV, ax=ax, label=label)
ax.legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'ZBC_low_H' + fig_format))
#%%


#%% conductance_before_after_ZBC
fig, axes = plt.subplots(2, 1, sharex=True)
Ns = ['0016', '0018', '0020']
amps = [1e10, 1e10, 1e10]
labels = ['H=0 before', 'H=0.12[T]', 'H=0 after']
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    _, _, _, time, _ = read_mesdata(os.path.join(data_dir + data_prefix + N + data_format))
    process_I_V(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, axes=axes, label=label+'-'+time[-8:])
axes[0].legend(loc='best')
#axes[1].legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'conductance_before_after_ZBC' + fig_format))
#%%
Ns = ['0016', '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029']
for N in Ns:
    try:
        sweeped, measured, data, measured_time, [T_min, T_max] = read_mesdata(os.path.join(data_dir + data_prefix + N + data_format))
        print(N, sweeped, measured, data.shape, measured_time, [T_min, T_max])
        print()
    except AssertionError:
        print(N)
#%% field cooling dIdV
        
fig, axes = plt.subplots(2, 1, sharex=True)
Ns = ['0016', '0051',] # can add 55, 56, 58, 62
amps = [1e10, 1e9]
labels = ['zero cooling', 'H=0.12[T] cooling']
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    process_I_V(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, axes=axes, label=label)
axes[0].legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'field_cooling_conductance' + fig_format))
#%% field cooling ZBC

fig = plt.figure()
ax = plt.subplot(111)
Ns = ['0017', '0019', '0053']
amps = [1e10, 1e10, 1e9]
labels = ['ZC up', 'ZC down', 'FC']
I_mins = [0, 0, 0,]
dVs = np.array([100e-6, 100e-6, 100e-6, 100e-6])
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    I_min = I_mins[i]
    dV = dVs[i]
    process_I_H(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, I_min=I_min, dV=dV, ax=ax, label=label)
ax.legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'ZBC_field_cooling' + fig_format))


#%%field cooling ZBC 1T
fig = plt.figure()
ax = plt.subplot(111)
Ns = ['0053', '0057', '0060', '0097']
amps = [1e10, 1e10, 1e10, 1e10]
labels = ['FC up 0.12T', 'FC up 1T', 'FC down', 'zero cooling']
I_mins = [0, 0, 0,0]
dVs = np.array([100e-6, 100e-6, 100e-6, 100e-6, 100e-6])
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    I_min = I_mins[i]
    dV = dVs[i]
    process_I_H(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, I_min=I_min, dV=dV, ax=ax, label=label)
ax.legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'ZBC_field_cooling_1T' + fig_format))


#%%
fig = plt.figure()
ax = plt.subplot(111)
process_record(os.path.join(data_dir + data_prefix + '0040' + data_format), amp=1e9, I_min=0, dV=100e-6, ax=ax, label='')
fig.savefig(os.path.join(figs_dir + 'record_40' + fig_format))


#%%recird bistability
fig = plt.figure()
ax = plt.subplot(111)
process_record(os.path.join(data_dir + data_prefix + '0111' + data_format), amp=1e10, I_min=0, dV=100e-6, ax=ax, label='')
fig.savefig(os.path.join(figs_dir + 'record_111' + fig_format))
ax.set_xlim(0, 300)
fig.savefig(os.path.join(figs_dir + 'record_111_zoom_300' + fig_format))
ax.set_xlim(0, 50)
fig.savefig(os.path.join(figs_dir + 'record_111_zoom_50' + fig_format))

#%%
sweeped, measured, data, measured_time, [T_min, T_max] = read_mesdata(os.path.join(data_dir + data_prefix + '0111' + data_format))
time, Idc = read_record(os.path.join(data_dir + data_prefix + '0111' + data_format), 1e10)
fig = plt.figure()
ax = fig.add_subplot(111)
_ = ax.plot(time[::2], Idc[::2], '.')
_ = ax.plot(time[1::2], Idc[1::2], '.')
#
#_ = ax.plot(time[::4], Idc[::4], '.')
#_ = ax.plot(time[1::4], Idc[1::4], '.')
#_ = ax.plot(time[2::4], Idc[2::4], '.')
#_ = ax.plot(time[3::4], Idc[3::4], '.')

#ax.set_xlim(0, 50)

#%%
for n in range(102, 143):
    sweeped, measured, data, measured_time, [T_min, T_max] = read_mesdata(os.path.join(data_dir + data_prefix + '0' + str(n) + data_format))
    print(n, measured_time, data.shape[0], sweeped)
#%%ZBC many end
fig = plt.figure()
ax = plt.subplot(111)
Ns = ['0129', '0130', '0131', '0132', '0133', '0134', '0135']
amps = 7 * [1e10, ]
labels = ['up', 'down', 'up', 'down', 'up', 'down', 'up', 'down', 'up']
I_mins = 7 * [0,]
dVs = np.array(7 * [100e-6,])
assert len(Ns) == len(amps)
for i in range(len(Ns)):
    N = Ns[i]
    amp = amps[i]
    label = labels[i]
    I_min = I_mins[i]
    dV = dVs[i]
    process_I_H(os.path.join(data_dir + data_prefix + N + data_format), amp=amp, I_min=I_min, dV=dV, ax=ax, label=label)
ax.legend(loc='best')
fig.savefig(os.path.join(figs_dir + 'ZBC_many' + fig_format))
    


#%%
plt.show()

if False:
    H = data[:,0]
    print(H.min())
    print(H.max())
    Vdc = data[:,1]
    print(Vdc.min(), Vdc.max())
    Vdc2 = data[:,2]
    
    
    print(data.min(axis=0))
    print(data.max(axis=0))

#plt.plot(H)


    


