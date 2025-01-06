import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import src.PYCCF as myccf
from src.pathlib import *

from scipy import stats

import sqlite3


# Plot Configurations

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


class LightCurve :
    """
    Light Curve tool for preKAMP database. 

	Attributes
	----------
	objname : str
		name of the source in the format of preKAMP Jname.
	data_dict : dict -> pandas.Dataframe
		dictionary containing lightcurves for each BVI bands.
	epochnum_dict : dict -> int
		dictionary containing number of epochs for each BVI light curves.

	Methods
	-------
	plot_lightcurve(mag_type='aperture')
		plots light curves for all BVI bands.
	calc_ccf(filt1, filt2)
		calculates cross-correlation function between light curves of filt1 and filt2.
  
    """

    def __init__(self, objname) -> None:
        """
        Initialize LightCurve by input object name.

		Parameters
  		----------
		objname : str
  			name of the source in the format of preKAMP Jname.
        
        """
        self.objname = objname
        self.data_dict = dict.fromkeys(filters)

        self._read()
        #self._aperture_photometry()
        #self._calc_epoch_num()
        pass


    def _read_init(self) -> list :
        """
        Check sql tables within sql DB. (DB created from pkamp-dataprocess)
        """

        conn = sqlite3.connect(LCDBPATH)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables_list = [i[0] for i in cursor.fetchall()]

        cursor.close()
        conn.close()

        return tables_list


    def _read(self) -> None :
        """
        Collect light curve data for corresponding object name via SQL query.
        """

        sql_tables_list = self._read_init()
        
        conn = sqlite3.connect(LCDBPATH)

        for filt in filters :

            tb_ls = [tb for tb in sql_tables_list if '_'+filt in tb]

            if len(tb_ls) > 0 :

                df_ls = []

                for tb_name in tb_ls :
                    sql_prompt = f"SELECT * FROM {tb_name} WHERE Jname = '{self.objname}'"
                    
                    df = pd.read_sql(sql_prompt, conn)
                    df_ls.append(df)

                self.data_dict[filt] = pd.concat(df_ls, ignore_index=True)
                
        conn.close()



    def plot_lightcurve(self, mag_type='aperture') -> None :
        """
        Plots light curve for object for BVI filters.
        """
        mag_type_dict = {
            'kron' : 'inst',
            'aperture' : 'gc'
        }

        fig,ax = plt.subplots(3,1, figsize=(15,10))

        colors_ls = ['b', 'g', 'r']
        ecolors_ls = ['skyblue', 'yellowgreen', 'magenta']

        mjd_min = 10**10
        mjd_max = 10**10

        for pltidx in range(3) :

            filt = filters[pltidx]
            data = self.data_dict[filt]
            
            if type(data) != type(None) :
                
                key = mag_type_dict[mag_type]
                mag_key = f'mag_{key}'
                magerr_key = f'magerr_{key}'

                err_lim = np.median(data[mag_key]) * 0.005
                lc_data = data[data[magerr_key] < err_lim]

                fwhm_med = np.median(lc_data['FWHM_IMAGE'])
                fwhm_std = np.std(lc_data['FWHM_IMAGE'])

                lc_data_cut = lc_data[(lc_data['FWHM_IMAGE'] < fwhm_med + fwhm_std) & (lc_data['FWHM_IMAGE'] > fwhm_med - fwhm_std)]
                lc_data_sort = lc_data_cut.sort_values('mjd')
                
                mjd_max = np.min([mjd_max, np.max(lc_data['mjd'])])
                mjd_min = np.min([mjd_min, np.min(lc_data['mjd'])])


                ax[pltidx].scatter(
                    x=lc_data['mjd'], y=lc_data[mag_key], edgecolor='none', c='gray', s=5
                )

                ax[pltidx].errorbar(
                    x=lc_data_sort['mjd'], y=lc_data_sort[mag_key], yerr=lc_data_sort[magerr_key],
                    fmt='o',
                    mfc=colors_ls[pltidx], mec='none', ecolor=ecolors_ls[pltidx], elinewidth=1, capsize=2, markersize=5
                )
                
            ax[pltidx].set_ylabel(filt)
            ax[pltidx].yaxis.label.set_color(colors_ls[pltidx])
            ax[pltidx].tick_params(axis='y', colors=colors_ls[pltidx])
                

        for pltidx in range(3) :    
            ax[pltidx].set_xlim(mjd_min-10, mjd_max+10)
            ax[pltidx].grid(alpha=0.3)
            ax[pltidx].invert_yaxis()
            
            if pltidx != 2 :
                ax[pltidx].set_xticklabels([])

        fig.supxlabel('MJD [days]')
        fig.supylabel('Brightness [mag]')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


    def _aperture_photometry(self) -> None :
        """
		Executes aperture photometry by navigating mag_aper(r=5*FWHM) of the source. Magnitude is calculated by linear interpolation of the growth curves.
  
  		Will update data_dict by adding the following columns.
  
  		'mag_gc' : aperture magnitude corresponding to 5*FWHM of the source.
		'magerr_gc' : corresponding uncertainty.
		"""
        for filt in filters :

            data = self.data_dict[filt]
            if type(data) != type(None) :

                mag_gc_ls = []
                magerr_gc_ls = []

                for idx in data.index : 
                    test_obj = data.loc[idx]
                    fwhm_5 = test_obj['seeing'] * 0.4 * 4

                    if fwhm_5 < 30 : 
                        fwhm_low_lim = int(np.floor(fwhm_5))
                        low_lim = f'MAG_APER_{fwhm_low_lim-1}'
                        mag_low_lim = test_obj[low_lim]
                        err_low_lim = test_obj[f'MAGERR_APER_{fwhm_low_lim-1}']

                        fwhm_upp_lim = int(np.ceil(fwhm_5))
                        upp_lim = f'MAG_APER_{fwhm_upp_lim-1}'
                        mag_upp_lim = test_obj[upp_lim]
                        err_upp_lim = test_obj[f'MAGERR_APER_{fwhm_upp_lim-1}']

                        if ((mag_upp_lim - mag_low_lim) != 0) & (mag_upp_lim < mag_low_lim) :

                            x1 = (mag_upp_lim - mag_low_lim) / (fwhm_upp_lim - fwhm_low_lim)
                            x0 = mag_upp_lim - x1*fwhm_upp_lim

                            mag_gc = x0 + x1*fwhm_5 + test_obj['delmag']
                            #magerr_gc = np.sqrt(err_low_lim**2 + err_upp_lim**2 + test_obj['magerr_inst']**2)
                            magerr_gc = (err_low_lim+err_upp_lim)/2
                        else :
                            mag_gc = 99
                            magerr_gc = 99
                    else :
                            mag_gc = 99
                            magerr_gc = 99

                    mag_gc_ls.append(mag_gc)
                    magerr_gc_ls.append(magerr_gc)

                data.insert(len(data.columns), "mag_gc", mag_gc_ls, True)
                data.insert(len(data.columns), "magerr_gc", magerr_gc_ls, True)

                self.data_dict[filt] = data

    def _calc_epoch_num(self) :
        """
		Calculates total epoch numbers for each BVI bands. 
  		"""

        self.epochnum_dict = dict.fromkeys(filters)

        for filt in filters :
            data = self.data_dict[filt]
            if len(data) == 0 :
                self.epochnum_dict[filt] = 0
            else :
                if type(data) != type(None) :
                    err_lim = np.median(data['mag_gc']) * 0.005
                    self.epochnum_dict[filt] = len(data[data['magerr_gc'] < err_lim])


    def _data4ccf(self, filt) :
        data = self.data_dict[filt]
        err_lim = np.median(data['mag_gc']) * 0.005
        lc_data = data[data['magerr_gc'] < err_lim]

        fwhm_med = np.median(lc_data['FWHM_IMAGE'])
        fwhm_std = np.std(lc_data['FWHM_IMAGE'])

        lc_data_cut = lc_data[(lc_data['FWHM_IMAGE'] < fwhm_med + fwhm_std) & (lc_data['FWHM_IMAGE'] > fwhm_med - fwhm_std)]
        lc_data_sort = lc_data_cut.sort_values('mjd')

        mjd = np.array(lc_data_sort['mjd'])
        mag = np.array(lc_data_sort['mag_gc'])
        magerr = np.array(lc_data_sort['magerr_gc'])
        
        return mjd, mag, magerr
    

    def calc_ccf(self, filt1, filt2, plot=True) :
        """
		Calculates most-likely time lag between light curves of filt1 and filt2. Source code directly adopted from PYCCF.

  		Parameters
		----------
  		filt1 : str
			Filter (band) of the first light curve.
   		filt2 : str
	 		Filter (band) pf the second light curve.
		plot : bool
  			TRUE to plot light curves, CCF, CCCD, and CCPD.

  		Returns
		-------
  		tuple
			a tuple of (centau, centau_uperr, centau_loerr, peaktau, peaktau_uperr, peaktau_loerr)
  		"""
        mjd1, mag1, err1 = self._data4ccf(filt1)
        mjd2, mag2, err2 = self._data4ccf(filt2)

        lag_range = [-100, 100]  #Time lag range to consider in the CCF (days). Must be small enough that there is some overlap between light curves at that shift (i.e., if the light curves span 80 days, these values must be less than 80 days)
        interp = 2. #Interpolation time step (days). Must be less than the average cadence of the observations, but too small will introduce noise.
        nsim = 500  #Number of Monte Carlo iterations for calculation of uncertainties
        mcmode = 0  #Do both FR/RSS sampling (1 = RSS only, 2 = FR only) 
        sigmode = 0.2  #Choose the threshold for considering a measurement "significant". sigmode = 0.2 will consider all CCFs with r_max <= 0.2 as "failed". See code for different sigmodes.


        tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_rval, status_rval, pval = myccf.peakcent(mjd1, mag1, mjd2, mag2, lag_range[0], lag_range[1], interp)
        tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals = myccf.xcor_mc(mjd1, mag1, abs(err1), mjd2, mag2, abs(err2), lag_range[0], lag_range[1], interp, nsim = nsim, mcmode=mcmode, sigmode = 0.2)

        lag = ccf_pack[1]
        r = ccf_pack[0]

        perclim = 84.1344746    

        ###Calculate the best peak and centroid and their uncertainties using the median of the
        ##distributions. 
        centau = stats.scoreatpercentile(tlags_centroid, 50)
        centau_uperr = (stats.scoreatpercentile(tlags_centroid, perclim))-centau
        centau_loerr = centau-(stats.scoreatpercentile(tlags_centroid, (100.-perclim)))
        print('Centroid, error: %10.3f  (+%10.3f -%10.3f)'%(centau, centau_uperr, centau_loerr))

        peaktau = stats.scoreatpercentile(tlags_peak, 50)
        peaktau_uperr = (stats.scoreatpercentile(tlags_peak, perclim))-centau
        peaktau_loerr = centau-(stats.scoreatpercentile(tlags_peak, (100.-perclim)))
        print('Peak, errors: %10.3f  (+%10.3f -%10.3f)'%(peaktau, peaktau_uperr, peaktau_loerr))

        if plot :
            fig = plt.figure(figsize=(15,12))
            fig.subplots_adjust(hspace=0.2, wspace = 0.1)

            #Plot lightcurves
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.errorbar(mjd1, mag1, yerr = err1, marker = '.', linestyle = ':', color = 'k', label = 'LC 1 (Continuum)')
            ax1_2 = fig.add_subplot(3, 1, 2, sharex = ax1)
            ax1_2.errorbar(mjd2, mag2, yerr = err2, marker = '.', linestyle = ':', color = 'k', label = 'LC 2 (Emission Line)')

            ax1.text(0.025, 0.825, "V band", fontsize = 15, transform = ax1.transAxes)
            ax1_2.text(0.025, 0.825, "I band", fontsize = 15, transform = ax1_2.transAxes)
            ax1.set_ylabel('LC 1 Flux')
            ax1_2.set_ylabel('LC 2 Flux')
            ax1_2.set_xlabel('MJD')

            #Plot CCF Information
            xmin, xmax = -99, 99
            ax2 = fig.add_subplot(3, 3, 7)
            ax2.set_ylabel('CCF r')
            ax2.text(0.2, 0.85, 'CCF ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax2.transAxes, fontsize = 16)
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(-1.0, 1.0)
            ax2.plot(lag, r, color = 'k')

            ax3 = fig.add_subplot(3, 3, 8, sharex = ax2)
            ax3.set_xlim(xmin, xmax)
            ax3.axes.get_yaxis().set_ticks([])
            ax3.set_xlabel('Centroid Lag: %5.1f (+%5.1f -%5.1f) days'%(centau, centau_uperr, centau_loerr), fontsize = 15) 
            ax3.text(0.2, 0.85, 'CCCD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax3.transAxes, fontsize = 16)
            n, bins, etc = ax3.hist(tlags_centroid, bins = 50, color = 'b')

            ax4 = fig.add_subplot(3, 3, 9, sharex = ax2)
            ax4.set_ylabel('N')
            ax4.yaxis.tick_right()
            ax4.yaxis.set_label_position('right') 
            #ax4.set_xlabel('Lag (days)')
            ax4.set_xlim(xmin, xmax)
            ax4.text(0.2, 0.85, 'CCPD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 16)
            ax4.hist(tlags_peak, bins = bins, color = 'b')

            plt.show()

        return centau, centau_uperr, centau_loerr, peaktau, peaktau_uperr, peaktau_loerr, [lag,r], tlags_centroid


    def calc_stats(self, filt1, filt2) :
        """
		Calculates CCF and median magnitude, both including associated uncertainties.

  		Parameters
		----------
  		filt1 : str
			Filter (band) of the first light curve.
   		filt2 : str
	 		Filter (band) of the second light curve.

 		Returns
   		-------
		df : pandas.DataFrame for the result statistics.
  		"""

        mjd1, mag1, err1 = self._data4ccf(filt1)
        mjd2, mag2, err2 = self._data4ccf(filt2)
        
        centau, centau_uperr, centau_loerr, peaktau, peaktau_uperr, peaktau_loerr = self.calc_ccf(filt1, filt2)

        df = pd.DataFrame([{
            'objname' : self.objname,
            'mag_med1' : np.median(mag1),
            'mag_std1' : np.sqrt(np.std(mag1)**2 + np.median(err1**2)**2),
            'mag_med2' : np.median(mag2),
            'mag_std2' : np.sqrt(np.std(mag2)**2 + np.median(err2**2)**2),
            'centau'   : centau,
            'centau_loerr' : centau_loerr,
            'centau_uperr' : centau_uperr,
            'peaktau' : peaktau,
            'peaktau_loerr' : peaktau_loerr,
            'peaktau_uperr' : peaktau_uperr,
        }])

        return df





#plot_lightcurve(df, 'a')


#test_obj = df.iloc[1767375]

 

#test_obj = df.iloc[1391400]

def plot_growthcurve(test_obj) :
    apersizes = np.array(range(1,31,1))*2.5 *0.4
    magapers = np.array([test_obj['MAG_APER']]+[test_obj[f'MAG_APER_{i}'] for i in range(1,30)])
    magapers_err = np.array([test_obj['MAGERR_APER']]+[test_obj[f'MAGERR_APER_{i}'] for i in range(1,30)])


    plt.errorbar(
        x=apersizes, y=magapers, yerr=magapers_err, c='b',
        ecolor='gray', elinewidth=1, capsize=2, marker='o', markersize=5, alpha=0.5,
        label='growth curve'
    )

    plt.axhline(test_obj['MAG_AUTO'], c='r', label='MAG_AUTO')
    plt.axvline(test_obj['seeing']*0.4*4, c='g', ls=':', label='4*seeing')

    plt.gca().invert_yaxis()

    plt.xlabel('Aperture size [arcsec]')
    plt.ylabel('mag')

    plt.legend(loc=4)
    plt.show()



    

