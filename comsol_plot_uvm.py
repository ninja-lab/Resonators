#%%
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import FormatStrFormatter

# df = pd.read_csv(r'Z:\data\Jackson_Anderson\MIDAS\FinFET\3D\MxTran_YM.txt',
#                  delimiter=r'\s\s*', skiprows=7)

#plt.style.use('seaborn-paper')
#plt.style.use(os.path.join(os.getcwd(), 'jackson.mplstyle'))

#dir = r'Z:\Jackson_Anderson\UVM\GaN_MEMS\comsol'

def read_comsol_txt_new(path, skiprows=0):
    '''
    Comsol Parser for Eigenfrequencies study default export with parameter, frequency, damping, Q
    '''
    pitch = []
    freq = []
    Q = []
    f = open(path)
    for i, l in enumerate(f):
        print(l)
        if i< skiprows:
            continue
        else:
            l = l.strip()
            l = re.sub('i', 'j', l)
            l = complex(l)
            if i % 5 == 0:
                pitch.append(l.real)
            elif i % 5 == 1:
                freq.append(l)
            elif i % 5 == 4:
                Q.append(l.real)


    return pd.DataFrame({
        'pitch': pitch,
        'Q': Q,
        'Eigenfrequency': freq
    })


df = pd.read_csv('gan_lamb_pitchsweep.txt',
    #os.path.join(dir,'gan_lamb_pitchsweep.txt'),
    sep='\s+',
    names=['pitch', 'Eigenfrequency', 'ang freq', 'damping ratio', 'Q'],
    skiprows=5
)

# https://stackoverflow.com/questions/18919699/python-pandas-complex-number
# df['X.8'].str.replace('i','j').apply(lambda x: np.complex(x))

df['Eigenfrequency'] = df['Eigenfrequency'].str.replace('i','j').apply(lambda x: complex(x))
df['real'] = np.real(df['Eigenfrequency'])
df['imag'] = np.imag(df['Eigenfrequency'])
df = df[df.Q > 0]

#%%

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
sc = ax3.scatter(df['pitch']*1E6, df['real'], marker='.',alpha=df['Q']/max(df['Q']))
ax3.yaxis.set_major_formatter(EngFormatter(unit='Hz'))
ax3.set_xscale('log')
#ax3.set_yscale('log')
# cm = plt.colorbar(sc)
# cm.set_label('log10(imag(f))')
#ax3.set_title('$L_{gate}$=106 nm fRBT')
#ax3.set_ylim([6E9, 10.5E9])
#ax3.get_legend().remove()
#ax3.set_xlabel(r'k$_x$ $[\pi/a]$')
ax3.set_xlabel(r'IDT Pitch [um]')
plt.tick_params(axis='y', which='minor')
#ax3.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
plt.tick_params(axis='x', which='minor')
ax3.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
ax3.set_ylabel(r'Frequency')
# ax3.yaxis.set_label_position("right")
# ax3.yaxis.tick_right()
# plt.gca().invert_xaxis()
# set_title(t)
plt.tight_layout()
plt.show()
