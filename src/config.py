import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# path settings
ROOTPATH = os.getcwd()
SRCEDIRC = os.path.join(ROOTPATH, 'src')

DATADIRC = os.path.join(ROOTPATH, 'data')
CACHDIRC = os.path.join(DATADIRC, 'cache')
ECATDIRC = os.path.join(DATADIRC, 'ext_catalogs')
PKATDIRC = os.path.join(DATADIRC, 'pkamp_catalogs')
#LCDBDIRC = os.path.join(DATADIRC, 'lightcurves')

BKUPDIRC = '/Volumes/SSD2TB/preKAMP/'
LCDBDIRC = os.path.join(BKUPDIRC, 'LAAKE_output')
LICVDIRC = os.path.join(LCDBDIRC, 'lightcurves')
LCPTDIRC = os.path.join(LCDBDIRC, 'lc_prop')



# plot settings
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 15}

mpl.rc('font', **font)

plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["axes.linewidth"] = 2


# auxiliary
filters = ['B', 'V', 'I']