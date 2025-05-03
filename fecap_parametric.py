#%%
import re
import os
import  skrf as rf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from lmfit import models


modes = {'15': {'freq': 704E6},
         '16': {'freq': 723E6},
         '17': {'freq': 721E6},
         '18': {'freq': 666E6},
}

# %% Plot S21

datadir = r'Z:\data\Measurement Data\FeCAP\2021\die1'

#dataregex = r'.*fecap_device15.*round2_600to800_power(-?\d*(p|_)\d*).*_.*gate(1_(1|2)).*V.s2p'

#dataregex = r'dev15_die3_TI_fecap_post2x100cyc_694to710MHzpower(-10(\.)0)_.*_gate(.*)V.s2p'
dataregex = r'die3_TI_fecap_device15_round2_600to800_power(-10())_.*_gate(.*)V.s2p'

files = os.listdir(datadir)
# filter files to snp only
files = [m.group(0) for f in files for m in [re.search(dataregex, f)] if m]

# sort files list by power, then voltage
files.sort(key=lambda x: float(re.sub(r'_|p', r'.', re.search(dataregex, x).group(1))))
files.sort(key=lambda x: float(re.sub(r'_', r'.', re.search(dataregex, x).group(3))))

X = []  # frequencies
Y = []  # input power or voltage
Z = []  # S21, db
for f in files:
    n = rf.Network(os.path.join(datadir, f))
    n['694Mhz-707MHz'].plot_y_db(m=1, n=0)  # S21, sb

    if not X:
        X.append(n.f)
    Y.append(re.sub(r'_', r'.', re.search(dataregex, f).group(3))) # 1 or 3, depending on voltage/power sweep
    Z.append(n.s21.y_db[:, 0, 0])


# Surface plot code

X = np.asfarray(X)
Y = np.asfarray(Y)
Z = np.asfarray(Z)

sortorder = np.argsort(Y, 0)
X, Y = np.meshgrid(X, Y)
[X, Y, Z] = [np.array(j)[sortorder] for j in [X, Y, Z]]

fig, ax1 = plt.subplots(1, 1, num=5, figsize=[9, 6])
#ax1 = fig.add_subplot(111)
cs = ax1.contourf(X, Y, Z, 50)#, cmap=cm.jet)
ax1.set_xlabel("frequency [Hz]")
ax1.set_ylabel("Bias [V]")
# ax1.set_ylabel("Power [dBm]")
#ax1.set_title(re.match(r'.*(device\d*).*', files[0]).group(1))  # group 1 or 3, depending on power or voltage sweep
cbar = plt.colorbar(cs)
cbar.set_label('Y21 Magnitude [dB]')
# #%%
#
# def mBVD_model(freq:rf.Frequency, cm, lm, rm, c1, c2, rg, rc1, rc2, name=''):
#     '''
#     Creates network for mBVD model of form
#
#     Port1 ---Rc1------Rs-----Ls----Cs--------Rc2-- Port2
#                   |                     |
#                   --------C1-----C2-----
#                               |
#                               Rg
#                               |
#                              Gnd
#
#     :param freq:
#     :param float cm: motional capacitance, Farads
#     :param float lm: motional inductance, Henrys
#     :param float rm: motional resistance, Ohms
#     :param float c1: port 1 electrical feedthrough capacitance, Farads
#     :param float c2: port 2 electrical feedthrough capaci   tance, Farads
#     :param float rg: gate resistance, Ohms
#     :param float rc1: port 1 contact resistance, Ohms
#     :param float rc2: port 2 contact resistance, Ohms
#     :return skrf.Network: Network representing mBVD model
#
#
#     '''
    tl_media = rf.DefinedGammaZ0(freq, z0=50, gamma=1j * freq.w / rf.c)
    Cm = tl_media.capacitor(cm, name='Cm')
    Lm = tl_media.inductor(lm, name='Lm')
    Rm = tl_media.resistor(rm, name='Rm')
    C1 = tl_media.capacitor(c1, name='C1')
    C2 = tl_media.capacitor(c2, name='C2')
    Rg = tl_media.resistor(rg, name='Rg')
    Rc1 = tl_media.resistor(rc1, name='Rc1')
    Rc2 = tl_media.resistor(rc2, name='Rc2')

    gnd = rf.Circuit.Ground(freq, name='gnd')
    port1 = rf.Circuit.Port(freq, name='port1', z0=50)
    port2 = rf.Circuit.Port(freq, name='port2', z0=50)

    cnx = [
        [(Rc1, 0),  (port1, 0)],
        [(Rc1, 1), (Cm, 0), (C1, 0)],
        [(Cm, 1), (Lm, 0)],
        [(Lm, 1), (Rm, 0)],
        [(C1, 1), (C2, 1), (Rg, 0)],
        [(gnd, 0), (Rg, 1)],
        [(Rm, 1), (Rc2, 1), (C2, 0)],
        [(Rc2, 0), (port2, 0)],
    ]
    cir = rf.Circuit(cnx)
    cir.plot_graph(network_labels=True, network_fontsize=15,
                    port_labels=True, port_fontsize=15,
                    edge_labels=True, edge_fontsize=10)
    ntw = cir.network
    ntw.name='mBVD_'+name
    return ntw
#
def fit_func(ntwk):
    # return ntwk.s[:, 1, 0]
    return np.concatenate((ntwk.s[:, 1, 0], ntwk.s[:, 0, 0],
                        ntwk.y[:, 1, 0], ntwk.y[:, 0, 0]))

def mBVD_fit_cost_function(ntwk:rf.Network, cm, fs, rm, c1, c2, rg, rc1, rc2):
    print('evaluating')
    cm = cm*1E-9
    lm = 1/(cm*(fs*2*np.pi)**2)
    model = mBVD_model(ntwk.frequency, cm, lm, rm, c1*1E-9, c2*1E-9, rg, rc1, rc2)
    return fit_func(ntwk)
#
# def mBVD_fit(ntwk:rf.Network, cm, fs, rm, c1, c2, rg, rc1, rc2, eta):
#     # model checks with 1E-13 tolerance - need to scale small numbers up
#     # rtot = rtot+1E-6 # exclude 0 to prevent singular admittance matrix
#     cm = cm*1E9
#     c1 = c1*1E9
#     c2 = c2*1E9
#     #ls = ls*1E9
#     model = models.Model(mBVD_fit_cost_function, independent_vars=['ntwk'])
#     model.set_param_hint('fs', min=0.97 * fs, max=1.03 * fs, value=fs, vary=False)
#     model.set_param_hint('cm', min=0.8*cm*eta**2, max=1.3*cm*eta**2, value=cm*eta**2, vary=True)
#  #   model.set_param_hint('lm', min=0.8*ls, max=1.3*ls, value=ls, vary=True)
#     model.set_param_hint('rm', min=0.5*rm/eta**2, max=2*rm/eta**2, value=rm/eta**2)
#     model.set_param_hint('c1', min=0.5*c1, max=2*c1, value=c1, vary=False)
#     model.set_param_hint('c2', min=0.5*c2, max=2*c2, value=c2, vary=False)
#     model.set_param_hint('rg', min=0.5*rg, max=2*rg, value=rg, vary=False)
#     model.set_param_hint('rc1', min=0.5*rc1, max=2*rc1, value=rc1, vary=False)
#     model.set_param_hint('rc2', min=0.5*rc2, max=2*rc2, value=rc2, vary=False)
#
#     ntwk_res = ntwk[f'{fs-15E6}-{fs+15E6} Hz']
#     print('starting mBVD fit')
#     output = model.fit(fit_func(ntwk_res), ntwk=ntwk_res, method='differential_evolution', max_nfev=75)
#     print(output.fit_report())
#     return mBVD_model(
#         ntwk.frequency,
#         cm=output.best_values['cm']*1E-9*eta**2,
#         #ls=output.best_values['ls']*1E-9,
#         lm=1/(output.best_values['cm'] * 1E-9*(2*np.pi*output.best_values['fs'])**2),
#         rm=output.best_values['rm']/eta**2,
#         c1=output.best_values['c1']*1E-9,
#         c2=output.best_values['c2']*1E-9,
#         rg=output.best_values['rg'],
#         rc1=output.best_values['rc1'],
#         rc2=output.best_values['rc2'],
#         name='lmfit'
#       ), output
#
#
# def mBVD_fit_2step(ntwk: rf.Network, cm, fs, rm, cp, eta):
#     # model checks with 1E-13 tolerance - need to scale small numbers up
#     # rtot = rtot+1E-6 # exclude 0 to prevent singular admittance matrix
#     cm = cm*1E9
#     cp = cp*1E9
#     # ls = ls*1E9
#     model_ft = models.Model(mBVD_fit_cost_function, independent_vars=['ntwk'])
#     model_ft.set_param_hint('fs', min=0.98*fs, max=1.02*fs, value=fs, vary=False)
#     model_ft.set_param_hint('cm', value=cm*eta**2, vary=False)
#     model_ft.set_param_hint('rm', value=1E6, vary=False)
#
#     model_ft.set_param_hint('c1', min=0.1*cp, max=10*cp, value=cp, vary=True)
#     model_ft.set_param_hint('c2', min=0.1*cp, max=10*cp, value=cp, vary=True)
#     model_ft.set_param_hint('rg', min=0.1, max=100, value=4.5, vary=True)
#     model_ft.set_param_hint('rc1', min=0.1, max=100, value=4.5, vary=True)
#     model_ft.set_param_hint('rc2', min=0.1, max=100, value=4.5, vary=True)
#     print('starting mBVD feedthrough fit')
#     ft_fit = model_ft.fit(fit_func(ntwk), ntwk=ntwk, method='differential_evolution', max_nfev=10)
#     print(ft_fit.fit_report())
#
#     model = models.Model(mBVD_fit_cost_function, independent_vars=['ntwk'])
#     model.set_param_hint('fs', min=0.98*fs, max=1.02*fs, value=fs, vary=True)
#     model.set_param_hint('cm', min=0.8*cm*eta**2, max=1.3*cm*eta**2, value=cm*eta**2,
#                          vary=True)
#     #   model.set_param_hint('lm', min=0.8*ls, max=1.3*ls, value=ls, vary=True)
#     model.set_param_hint('rm', min=0.1*rm/eta**2, max=10*rm/eta**2, value=rm/eta**2)
#     model.set_param_hint('c1', min=0.5*cp, max=2*cp, value=ft_fit.best_values['c1'], vary=False)
#     model.set_param_hint('c2', min=0.5*cp, max=2*cp, value=ft_fit.best_values['c2'], vary=False)
#     model.set_param_hint('rg', min=0.1, max=10, value=ft_fit.best_values['rg'], vary=False)
#     model.set_param_hint('rc1', min=0.1, max=10, value=ft_fit.best_values['rc1'], vary=False)
#     model.set_param_hint('rc2', min=0.1, max=10, value=ft_fit.best_values['rc2'], vary=False)
#
#     ntwk_res = ntwk[f'{fs-5E6}-{fs+5E6} Hz']
#     print('starting mBVD resonance fit')
#     output = model.fit(fit_func(ntwk_res), ntwk=ntwk_res,
#                        method='basinhopping', max_nfev=10)
#     print(output.fit_report())
#     return mBVD_model(
#         ntwk.frequency,
#         cm=output.best_values['cm']*1E-9*eta**2,
#         # ls=output.best_values['ls']*1E-9,
#         lm=1/(output.best_values['cm']*1E-9*(2*np.pi*output.best_values['fs'])**2),
#         rm=output.best_values['rm']/eta**2,
#         c1=output.best_values['c1']*1E-9,
#         c2=output.best_values['c2']*1E-9,
#         rg=output.best_values['rg'],
#         rc1=output.best_values['rc1'],
#         rc2=output.best_values['rc2'],
#         name='lmfit'
#     ), output
#
def gauss_fit(ntwk: rf.Network, fs):

    y = np.abs(ntwk.s[:, 1, 0])
    sigma_max = 10E6
    model = models.GaussianModel() + models.LinearModel()
    model.set_param_hint('center', value=fs)
    model.set_param_hint('sigma', value=sigma_max)
    model.set_param_hint('amplitude', value=np.max(y) * np.sqrt(np.pi * 2) * sigma_max)
    model.set_param_hint('intercept', value=0.07)
    model.set_param_hint('slope', value=0)

    output = model.fit(y, x=ntwk.f, max_nfev=250)
    print(output.fit_report())
    return output

def open_deembed(data, data_open, datadir=None):
    '''
    Subtracts admittance of open from that of device and saves results in new
    network n. If save_deembedded is defined, saves a copy of the deembedded
    network in snp format.
    '''
    d = d = rf.Network(data)
    o = d = rf.Network(data_open)
    n = rf.Network()
    n.frequency = d.frequency
    n.z0 = d.z0
    n.name = d.name
    n.y = d.y - o.y
    n.s = rf.y2s(n.y)
    n.z = rf.y2z(n.y)
    if datadir:
        savedir = os.path.join(datadir, 'deembedded_open')
        if not os.path.isfile(os.path.join(savedir, n.name)):
            n.write_touchstone(dir=savedir)
    return n

def eta(V, Vc, Pr, Ps):
    '''
    Normalized ferroelectric polarization half-loop using tanh model.

    Ref: Lue et al, "Device Modeling of Ferroelectric Memory Field-Effect Transistor for the
    Application of Ferroelectric Random Access Memory," IEEE TUFFC, 2003

    :param float V: applied voltage
    :param float Vc: coercive voltage
    :param float Pr: remnant polarization after voltage removed
    :param float Ps: fully saturated polarization

    :returns Psat/Ps: normalized tanh polarization curve
    '''
    delta = Vc*np.log((1+Pr/Ps)/(1-Pr/Ps))**-1
    Psat = Ps*np.tanh((V/Vc)/(2*delta))
    return Psat/Ps

datadir = r'Z:\data\Measurement Data\FeCAP\2021\die1'
fitdir = os.path.join(datadir,'fitting')
dataregex = r'die3_TI_fecap_device(\d*).*round2_600to800_power(-?\d*(p|_)?\d*)_(\d*).*_.*gate(.*)V\.s2p'
open = ''

datadict = {
    'dataname': [],
    'datafile': [],
    'Device': [],
    'Power': [],
    'Vg': [],
    'f0': [],  # constant for ADS optimization goal f limits
    'fs': [],  # fs variable for ADS optimization
    'Q': [],
    'Rg': [],
    'R1t': [],
    'R2t': [],
    'Rc1': [],
    'Rc2': [],
    'C1': [],
    'C2': [],
    'Cm': [],
    'Rm': [],
    'Lm': [],
}

files = os.listdir(datadir)
# filter files to snp only
files = [m.group(0) for f in files for m in [re.search(dataregex, f)] if m]

for f in files:
    n = rf.Network(os.path.join(datadir,f))
    #no = rf.Network(datadir+r'\die3_TI_fecap_open_round2_600to800_power0_11_gate1_5V.s2p')
    # n = open_deembed(
    #         datadir+r'\die3_TI_fecap_device16_round2_600to800_power0_11_gate1_5V.s2p',
    #         datadir+r'\die3_TI_fecap_open_round2_600to800_power0_11_gate1_5V.s2p',
    #         datadir)
    m = re.search(dataregex, f)
    f0_guess = modes[m.group(1)]['freq']

    # plt.figure()
    # n.plot_s_mag(1,0)
    out = gauss_fit(n, f0_guess)
    Q = abs(out.best_values['center']/out.best_values['sigma']/2.3548200)
    Vg = float(re.sub(r'(_|p)', r'.', m.group(5)))
    # plt.plot(n.f, out.best_fit, label=f"fit, f0={out.best_values['center']:.4} Hz, Q={Q:.4}")
    # plt.legend()

    # if (out.best_values['center']-f0_guess) < 2:
    #     f0_guess = out.best_values['center']
    calc_freq = f0_guess+30E6
    rg_guess = np.real(n[f'{calc_freq}'].z[:, 1, 0])[0]
    r1_guess = np.real(n[f'{calc_freq}'].z[:, 0, 0])[0] - rg_guess
    r2_guess = np.real(n[f'{calc_freq}'].z[:, 1, 1])[0] - rg_guess
    c1_guess = np.abs(1/(np.imag(n[f'{calc_freq}'].z[:, 0, 0])*2*np.pi*calc_freq))[0]
    c2_guess = np.abs(1/(np.imag(n[f'{calc_freq}'].z[:, 1, 1])*2*np.pi*calc_freq))[0]
    cm_guess = 0.8E-15*max(eta(Vg, 0.29, 15, 37)**2, 0.01)
    rm_guess = 1/(cm_guess*2*np.pi*f0_guess*Q)-r1_guess-r2_guess

    lm_init = (2*np.pi*f0_guess)**-2/cm_guess

    datadict['dataname'].append(n.name)
    datadict['datafile'].append(os.path.join(datadir,f))
    datadict['Device'].append(m.group(1))
    datadict['Power'].append(float(re.sub(r'(_|p)', r'.', m.group(2))))
    datadict['Vg'].append(Vg)
    datadict['f0'].append(f0_guess)
    datadict['fs'].append(f0_guess)
    datadict['Q'].append(Q)
    datadict['Rg'].append(rg_guess)
    datadict['R1t'].append(r1_guess)
    datadict['R2t'].append(r2_guess)
    datadict['Rc1'].append(r1_guess/2)
    datadict['Rc2'].append(r2_guess/2)
    datadict['C1'].append(c1_guess)
    datadict['C2'].append(c2_guess)
    datadict['Cm'].append(cm_guess)
    datadict['Rm'].append(rm_guess)
    datadict['Lm'].append(lm_init)




    # nfit, model = mBVD_fit(n, cm_guess, f0_guess, rm_guess, c1_guess, c2_guess,
    #                        rg_guess, r1_guess, r2_guess, -1)
    #npub = mBVD_model(n.frequency, 0.85E-15, 57.4E-6, 374, 4.47E-12, 4.51E-12, 3.9, 4.43, 5.48, name='He2019')
    plot = 0
    if plot:
        nctrl = mBVD_model(n.frequency, cm_guess, lm_init, rm_guess, c1_guess, c2_guess,
                            rg_guess, r1_guess, r2_guess, name='Initial Guess')

        plt.figure()
        ntwk_name = n.name
        plt.title = n.name
        n.name = 'Measured'
        n[f'{f0_guess-15E6}-{f0_guess+10E6} Hz'].plot_s_db()
        # nfit.plot_s_db()
        nctrl[f'{f0_guess-15E6}-{f0_guess+10E6} Hz'].plot_s_db(ls='--')
        plt.legend(loc='center left')
        # npub.plot_s_db(ls='--')
        plt.savefig(os.path.join(fitdir, ntwk_name+'.png'))

df = pd.DataFrame.from_dict(datadict)
for d in ['15', '16', '17', '18']:
    df[df.Device == d].to_csv(os.path.join(fitdir, f'{d}_initial_guess.csv'), index=False)


#%%
ads_dir = r'C:\Users\ander906\Desktop\ADS\fecap_wrk\data'
varnames = ['dataname','Power','Vg','c1','c2','cm','fs','r1t','r2t','rc1','rc2','rg','rm']
df3 = pd.read_csv(os.path.join(ads_dir, r'device15_optmvals2.csv'), skiprows=1, names=varnames)

#df3 = df2.join(df[df.Device == '15'].reset_index().loc[: ,['dataname','Power','Vg']])
df3.sort_values(['Power', 'Vg'], inplace=True)

pd.plotting.scatter_matrix(df3)

figc = plt.figure()
axc = figc.add_subplot(111)
figp = plt.figure()
axp = figp.add_subplot(111)
figp1 = plt.figure()
axp1 = figp1.add_subplot(111)
figr = plt.figure()
axr = figr.add_subplot(111)
figrc = plt.figure()
axrc = figrc.add_subplot(111)
figrm = plt.figure()
axrm = figrm.add_subplot(111)
figcm = plt.figure()
axcm = figcm.add_subplot(111)
figrg = plt.figure()
axrg = figrg.add_subplot(111)

axp1.plot(df3.Power, df3.fs, ls='None', marker='.', label=f'fs {p}dBm')
axp1.set_xlabel('Power [dBm]')
axp1.set_ylabel('Resonant Frequency [Hz]')
axp1.set_title('Device 15')
for p in df3['Power'].unique():
    df4 = df3[df3.Power == p]
    # pd.plotting.scatter_matrix(df4, sharex=True, sharey=True)
    axc.plot(df4.Vg, df4.c1, ls='--', marker='.', label=f'C1 {p}dBm')
    axc.plot(df4.Vg, df4.c2, ls='--', marker='.', label=f'C2 {p}dBm')
    axc.set_ylabel('Capacitance [F]')
    axp.plot(df4.Vg, df4.fs, ls='--', marker='.', label=f'fs {p}dBm')
    axp.set_ylabel('Frequency [Hz]')
    axr.plot(df4.Vg, df4.r1t, ls='--', marker='.', label=f'R1t {p}dBm')
    axr.set_ylabel('Resistance [Ohm]')
    axr.plot(df4.Vg, df4.r2t, ls='--', marker='.', label=f'R2t {p}dBm')
    axr.set_ylabel('Resistance [Ohm]')
    axrg.plot(df4.Vg, df4.rg, ls='--', marker='.', label=f'Rg {p}dBm')
    axrg.set_ylabel('Resistance [Ohm]')
    axrm.plot(df4.Vg, df4.rm, ls='--', marker='.', label=f'Rm {p}dBm')
    axrm.set_ylabel('Resistance [Ohm]')
    axcm.plot(df4.Vg, df4.cm, ls='--', marker='.', label=f'Cm {p}dBm')
    axcm.set_ylabel('Capacitance [F]')

    axcm.plot(df4.Vg, 8E-16*eta(df4.Vg, 0.29, 15, 37)**2)

    axlm.plot(df4.Vg, df4.lm, ls='--', marker='.', label=f'Lm {p}dBm')
    axlm.set_ylabel('Inductance [H]')

for vg in df3['Vg'].unique():
    df4 = df3[df3.Vg == vg]
    axc.plot(df4.Power, df4.c1, ls='--', marker='.', label=f'C1 {vg} Volts')
    axc.plot(df4.Power, df4.c2, ls='--', marker='.', label=f'C2 {vg} Volts')
    axc.set_ylabel('Capacitance [F]')
    axp.plot(df4.Power, df4.fs, ls='--', marker='.', label=f'fs {vg} Volts')
    axp.set_ylabel('Frequency [Hz]')
    axr.plot(df4.Power, df4.r1t, ls='--', marker='.', label=f'R1t {vg} Volts')
    axr.plot(df4.Power, df4.r2t, ls='--', marker='.', label=f'R2t {vg} Volts')
    axr.set_ylabel('Resistance [Ohm]')
    axrc.plot(df4.Power, df4.rc1, ls='--', marker='.', label=f'Rc1 {vg} Volts')
    axrc.plot(df4.Power, df4.rc2, ls='--', marker='.', label=f'Rc2 {vg} Volts')
    axrc.set_ylabel('Resistance [Ohm]')
    axrg.plot(df4.Power, df4.rg, ls='--', marker='.', label=f'Rg {vg} Volts')
    axrg.set_ylabel('Resistance [Ohm]')
    axrm.plot(df4.Power, df4.rm, ls='--', marker='.', label=f'Rm {vg} Volts')
    axrm.set_ylabel('Resistance [Ohm]')
    axcm.plot(df4.Power, df4.cm, ls='--', marker='.', label=f'Cm {vg} Volts')

for ax in [axc, axp, axr, axrc, axrm, axcm, axrg]:
    ax.legend()
    # ax.set_xlabel('Gate Bias [V]')
    ax.set_xlabel('Power [dBm]')
    ax.set_title('Device 16')

#%%
#
# Muhammad A. Wahab, Purdue University, 2015 (Advisor: Prof. Muhammad A. Alam)
# NC Material                          alphaFE (cm/F)     betaFE (cm^5/F/Coul^2)     gammaFE (cm^9/F/Coul^4)    Ref  %%%
# BaTiO_3                                -5e8                -2.2250e18                  7.5e27                  [1]-[2]
# PZT (PbZr_{1-x}Ti_{x}O_3)              -2.25e9              1.3e18                     9.8333e25               [1]
# SBT (Sr_{0.8}Bi_{2.2}Ta_2O_9)          -3.25e9              9.375e18                     0                     [1]
# P(VDF-TrFE)                            -1.8e11              5.8e22                       0                     [3]-[5]
# HfSiO                                  -8.65e10             1.92e20                      0                     [6]

#

a = -2.25e9
b = 1.3e18
g = 9.8333e25
P = np.linspace(-38E-6,38E-6,101)
dP = P[1]-P[0]
U = a*P**2+b*P**4+g*P**6
C = np.gradient(np.gradient(U, dP), dP)**-1
plt.figure()
plt.plot(P,C)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

# vals = [0.5, 0.7, 1, 1.2]
# for T in [x*T0 for x in vals]:
#vals = np.linspace(-1,1) # [-0.5, -0.2, 0, 0.2, 0.5]
#for E in [x for x in vals]:

U = a*P**2+b*P**4+g*P**6
E = np.gradient(U)
C = np.gradient(E)**-1

ind = np.append((np.diff(np.sign(E)) != 0),False) & (C > 0)


E2 = np.concatenate([E, E[::-1]])
P2 = np.concatenate([P, P[::-1]])
for i, p in enumerate(P2):
    if i == 0:
        continue
    elif np.sign(E2[i]-E2[i-1]) == np.sign(p-P2[i-1]):
        continue
    else:
        E2[i] = E2[i-1]

C2 = np.diff(P2)/np.diff(E2)

with plt.rc_context({'lines.linewidth':8}):
    # plt.figure()
    # plt.plot(P, E)
    ax1.plot(P2[0:-1], C2, lw=0, marker='.')
    #ax1.plot(E2, np.gradient(P2, E2)**-1)
    #plt.axis('off')
    ax2.plot(E2, P2)
    # ax2.set_xlim([-1, 1])
    # ax2.set_ylim([-1, 1])
    # plt.axis('off')