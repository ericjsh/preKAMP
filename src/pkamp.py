from src import config
from src.drw_utils import MCMC_fit

import os
import json
import numpy as np
import pandas as pd
import sqlite3

import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import match_coordinates_sky, SkyCoord
from astroquery.vizier import Vizier

class AgnCatalogs :
    
    def __init__(self, fieldname:str) -> None:
        self.fieldname = fieldname
        self._read_pkamp()
        self.match_cat_init()
        pass

    def _read_pkamp(self) -> None :
        pkamp_cat_fname = f'matched_targets_{self.fieldname}.csv'
        pkamp_cat_fpath = os.path.join(config.PKATDIRC, pkamp_cat_fname)

        self.pkamp_cat:Table = ascii.read(pkamp_cat_fpath)
        self.pkamp_cat.remove_column('sep')
        self.cent_ra:float = np.median(self.pkamp_cat['ra'])
        self.cent_dec:float = np.median(self.pkamp_cat['dec'])
        pass

    def _load_cat(self, catname:str, srch_radius:float=2) -> Table :

        local_catname = f'{catname}_{self.fieldname}.csv'
        cat_fpath:str = os.path.join(config.ECATDIRC, local_catname)

        if not os.path.isfile(cat_fpath) :

            ecat_vizier_fname:str = 'ecat_vizier.json'
            ecat_vizier_fpath:str = os.path.join(config.SRCEDIRC, ecat_vizier_fname)
            with open(ecat_vizier_fpath) as f :
                ecat_dict:dict = json.load(f)

            catalog_id:str = ecat_dict[catname]['vizier_id']

            Vizier.ROW_LIMIT = -1

            cat_query = Vizier.query_region(
                coord.SkyCoord(
                    ra=self.cent_ra, dec=self.cent_dec, 
                    unit=(u.deg, u.deg), frame='icrs'
                ),
                radius=srch_radius*u.deg,
                catalog=[catalog_id],
            )

            cat = cat_query[0]

            ascii.write(cat, cat_fpath, overwrite=True)

        cat = ascii.read(cat_fpath)

        return cat
    
    def match_cat_init(self) :

        self.catalog = self.pkamp_cat
        self.matched_catalogs = []

    def _extcat_radec_keys(self, ext_cat) :
        ra_key = [k for k in ext_cat.keys() if k.startswith('ra') or k.startswith('RA')][0]
        dec_key = [k for k in ext_cat.keys() if k.startswith('dec') or k.startswith('DE')][0]
        return ra_key, dec_key
    
    def match_cat(self, catname:str, srch_radius:float=2, mtch_radius:float=2) -> Table :

        self.matched_catalogs.append(catname)
        ext_cat = self._load_cat(catname=catname, srch_radius=srch_radius)
        ra_key, dec_key = self._extcat_radec_keys(ext_cat)
        
        pkamp_cat_coord = SkyCoord(
            ra=self.catalog['ra']*u.deg,
            dec=self.catalog['dec']*u.deg
        )

        ext_cat_coord = SkyCoord(
            ra=ext_cat[ra_key]*u.deg,
            dec=ext_cat[dec_key]*u.deg
        )

        mcat = self.catalog.copy()
        idx, d2d, d3d = pkamp_cat_coord.match_to_catalog_sky(ext_cat_coord)
        mcat['sep'] = d2d.to(u.arcsec)
        match_sucess = mcat[np.where(mcat['sep'] < mtch_radius*u.arcsec)]
        midx = idx[np.where(mcat['sep'] < 2*u.arcsec)]

        for key in ext_cat.keys() :
            new_keyname = f'{catname}_{key}'
            match_sucess[new_keyname] = ext_cat[midx][key]

        self.catalog = match_sucess

    def color_color_selection(self, plot=False) :

        self.match_cat_init()
        self.match_cat('gaiadr3')
        self.match_cat('allwise')
        kmtn_gaiadr3_allwise = self.catalog.copy()

        self.match_cat_init()
        self.match_cat('gaiadr3')
        self.match_cat('allwise')
        self.match_cat('gaiaqsocand')
        kmtn_gaiadr3_qsocand = self.catalog.copy()

        agn_color_cut = kmtn_gaiadr3_qsocand[np.where(
            (kmtn_gaiadr3_qsocand['allwise_W1mag'] - kmtn_gaiadr3_qsocand['allwise_W2mag'] > 0.4) &
            (kmtn_gaiadr3_qsocand['gaiadr3_Gmag'] - kmtn_gaiadr3_qsocand['allwise_W1mag'] > 2.15) &
            ((kmtn_gaiadr3_qsocand['gaiadr3_Gmag'] - kmtn_gaiadr3_qsocand['allwise_W1mag']) + 1.2*(kmtn_gaiadr3_qsocand['allwise_W1mag'] - kmtn_gaiadr3_qsocand['allwise_W2mag']) > 3.4) &
            (kmtn_gaiadr3_qsocand['gaiadr3_BPmag'] - kmtn_gaiadr3_qsocand['gaiadr3_Gmag'] > -0.3) &
            (0.315*(kmtn_gaiadr3_qsocand['allwise_W2mag']-kmtn_gaiadr3_qsocand['allwise_W3mag']) + 0.796 > kmtn_gaiadr3_qsocand['allwise_W1mag']-kmtn_gaiadr3_qsocand['allwise_W2mag']) &
            (0.315*(kmtn_gaiadr3_qsocand['allwise_W2mag']-kmtn_gaiadr3_qsocand['allwise_W3mag']) -0.222 < kmtn_gaiadr3_qsocand['allwise_W1mag']-kmtn_gaiadr3_qsocand['allwise_W2mag']) &
            (-3.172*(kmtn_gaiadr3_qsocand['allwise_W2mag']-kmtn_gaiadr3_qsocand['allwise_W3mag']) + 7.624 < kmtn_gaiadr3_qsocand['allwise_W1mag']-kmtn_gaiadr3_qsocand['allwise_W2mag'])
        )]

        if plot:
            def y_p(x) :
                return 0.315*x + 0.796

            def y_m(x) :
                return 0.315*x -0.222

            x_p = (7.624-0.796)/(0.315+3.172)
            x_m = (7.624+0.222)/(0.315+3.172)

            fig, ax = plt.subplots(1,3, figsize=(18,6))

            # Left panel
            ax[0].scatter(
                x=kmtn_gaiadr3_allwise['gaiadr3_Gmag'] - kmtn_gaiadr3_allwise['allwise_W1mag'], y=kmtn_gaiadr3_allwise['allwise_W1mag'] - kmtn_gaiadr3_allwise['allwise_W2mag'], 
                c='k', s=5, alpha=0.2, edgecolor='none', zorder=0, label=f'preKAMP-{self.fieldname} All sources'
            )

            ax[0].scatter(
                x=kmtn_gaiadr3_qsocand['gaiadr3_Gmag'] - kmtn_gaiadr3_qsocand['allwise_W1mag'], y=kmtn_gaiadr3_qsocand['allwise_W1mag'] - kmtn_gaiadr3_qsocand['allwise_W2mag'], 
                c='b', s=10, zorder=1, label=fr'preKAMP-{self.fieldname} $\times$ GAIA qsocand', alpha=0.3
            )

            ax[0].scatter(
                x=agn_color_cut['gaiadr3_Gmag'] - agn_color_cut['allwise_W1mag'], y=agn_color_cut['allwise_W1mag'] - agn_color_cut['allwise_W2mag'], 
                c='r', s=15, zorder=2, label=f'preKAMP-{self.fieldname} Type I AGN'
            )



            xp = np.linspace(2.15,3.4 - 1.2*0.4,100)
            yp = (3.4 - xp)/1.2
            ax[0].fill_between(xp, yp, 0.4, interpolate=True, color='gray', alpha=0.3, edgecolor='none')
            ax[0].fill_betweenx(np.linspace(-5,5), -5,2.15, interpolate=True, color='gray', alpha=0.3, edgecolor='none')
            ax[0].fill_betweenx(np.linspace(-5,0.4), 2.15,10, interpolate=True, color='gray', alpha=0.3, edgecolor='none')

            ax[0].set_xlim(-1,9)
            ax[0].set_ylim(-1,3)

            ax[0].plot([2.15, 2.15], [10, (3.4-2.15)/1.2], c='k')
            ax[0].plot([2.15, 3.4 - 1.2*0.4], [(3.4-2.15)/1.2, 0.4], c='k')
            ax[0].plot([3.4 - 1.2*0.4, 20], [0.4, 0.4], c='k')


            ax[0].set_xlabel('G $-$ W1')
            ax[0].set_ylabel('W1 $-$ W2')

            ax[0].legend()

            # Middele panel
            ax[1].scatter(
                x=kmtn_gaiadr3_allwise['gaiadr3_BPmag'] - kmtn_gaiadr3_allwise['gaiadr3_Gmag'], y=kmtn_gaiadr3_allwise['gaiadr3_Gmag'] - kmtn_gaiadr3_allwise['gaiadr3_RPmag'], 
                c='k', s=5, alpha=0.2, edgecolor='none', zorder=0
            )

            ax[1].scatter(
                x=kmtn_gaiadr3_qsocand['gaiadr3_BPmag'] - kmtn_gaiadr3_qsocand['gaiadr3_Gmag'], y=kmtn_gaiadr3_qsocand['gaiadr3_Gmag'] - kmtn_gaiadr3_qsocand['gaiadr3_RPmag'], 
                s=10, c='b', alpha=0.5
            )

            ax[1].scatter(
                x=agn_color_cut['gaiadr3_BPmag'] - agn_color_cut['gaiadr3_Gmag'], y=agn_color_cut['gaiadr3_Gmag'] - agn_color_cut['gaiadr3_RPmag'], 
                c='r', s=15
            )

            ax[1].fill_betweenx(np.linspace(-5,6), -5, -0.3, interpolate=True, color='gray', alpha=0.3)

            ax[1].axvline(-0.3,c='k')

            ax[1].set_xlabel('BP $-$ G')
            ax[1].set_ylabel('G $-$ RP')

            ax[1].set_xlim(-4.1, 2.9)
            ax[1].set_ylim(-1,5)



            ax[2].scatter(
                x=kmtn_gaiadr3_allwise['allwise_W2mag']-kmtn_gaiadr3_allwise['allwise_W3mag'], y=kmtn_gaiadr3_allwise['allwise_W1mag']-kmtn_gaiadr3_allwise['allwise_W2mag'], c='k', s=5, alpha=0.2, edgecolor='none'
            )

            ax[2].scatter(
                x=kmtn_gaiadr3_qsocand['allwise_W2mag']-kmtn_gaiadr3_qsocand['allwise_W3mag'], y=kmtn_gaiadr3_qsocand['allwise_W1mag']-kmtn_gaiadr3_qsocand['allwise_W2mag'], 
                c='b', s=10, alpha=0.3
            )

            ax[2].scatter(
                x=agn_color_cut['allwise_W2mag']-agn_color_cut['allwise_W3mag'], y=agn_color_cut['allwise_W1mag']-agn_color_cut['allwise_W2mag'], c='r', s=15
            )

            xp1 = np.linspace(x_p,10,100)
            xp2 = np.linspace(x_p, x_m)
            xp3 = np.linspace(x_m, 10, 100)
            yp2 = -3.172*xp2+7.624
            ax[2].fill_between(xp1, y_p(xp1), 10, interpolate=True, color='gray', alpha=0.3, edgecolor='none')
            ax[2].fill_between(xp2, yp2, y_m(x_m), interpolate=True, color='gray', alpha=0.3, edgecolor='none')
            ax[2].fill_between(xp3, y_m(xp3), y_m(x_m), interpolate=True, color='gray', alpha=0.3, edgecolor='none')
            ax[2].fill_betweenx(np.linspace(-5,5), -5, x_p, interpolate=True, color='gray', alpha=0.3, edgecolor='none')
            ax[2].fill_betweenx(np.linspace(-5,y_m(x_m)), x_p, 10, interpolate=True, color='gray', alpha=0.3, edgecolor='none')

            ax[2].plot([x_p, x_m], [y_p(x_p), y_m(x_m)], c='k')
            ax[2].plot([x_p, 10], [y_p(x_p), y_p(10)], c='k')
            ax[2].plot([x_m, 10], [y_m(x_m), y_m(10)], c='k')

            ax[2].set_xlabel('W2 $-$ W3')
            ax[2].set_ylabel('W1 $-$ W2')

            ax[2].set_xlim(-0.1,5.1)
            ax[2].set_ylim(-0.6, 2.1)

            plt.tight_layout()
            plt.show()

        return agn_color_cut




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

    def __init__(self, objname, field) -> None:
        """
        Initialize LightCurve by input object name.

		Parameters
  		----------
		objname : str
  			name of the source in the format of preKAMP Jname.
        field :
            name of the field the source is located at.
        """
        self.objname = objname
        self.field = field
        self.data_dict = dict.fromkeys(['B', 'V', 'I'])

        self._read_lightcurve()
        pass


    def _read_from_sql_init(self, db_fpath) -> list :
        """
        Check sql tables within sql DB. (DB created from pkamp-dataprocess)
        """

        conn = sqlite3.connect(db_fpath)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables_list = [i[0] for i in cursor.fetchall()]

        cursor.close()
        conn.close()

        return tables_list


    def _read_from_sql(self) -> None :
        """
        Collect light curve data for corresponding object name via SQL query.
        """

        if self.field == 'N55' :

            db_fname = 'N55_CTIO.db'
            db_fpath = os.path.join(config.LCDBDIRC, db_fname)
            sql_tables_list = self._read_from_sql_init(db_fpath)
        
            conn = sqlite3.connect(db_fpath)

            for filt in ['B', 'V', 'I'] :

                tb_ls = [tb for tb in sql_tables_list if '_'+filt in tb]

                if len(tb_ls) > 0 :

                    df_ls = []

                    for tb_name in tb_ls :
                        sql_prompt = f"SELECT * FROM {tb_name} WHERE Jname = '{self.objname}'"
                        
                        df = pd.read_sql(sql_prompt, conn)
                        df_ls.append(df)

                    self.data_dict[filt] = pd.concat(df_ls, ignore_index=True)
                
            conn.close()

        else :
            df_dict = {
                'B': [],
                'V': [],
                'I': []
            }

            db_fdir = '/Volumes/SSD2TB/preKAMP/LAAKE_output/'
            fieldname = 'N59'
            db_fnames = [f for f in os.listdir(db_fdir) if f.startswith(fieldname) and f.endswith('.db')]

            for db_fname in db_fnames :
                db_fpath = os.path.join(db_fdir, db_fname)
                tables_list = self._read_from_sql_init(db_fpath)
                    
                conn = sqlite3.connect(db_fpath)
                cursor = conn.cursor()
                
                for filt in ['B', 'V', 'I'] :
                
                    tb_name = 'CTIO_' + filt 
                
                    if tb_name in tables_list :
                        sql_prompt = f"SELECT * FROM {tb_name} WHERE Jname = '{self.objname}'"
                        
                        df = pd.read_sql(sql_prompt, conn)
                        df_dict[filt].append(df)
                
                cursor.close()
                conn.close()

            for filt in ['B', 'V', 'I'] :
                self.data_dict[filt] = pd.concat(df_dict[filt])


    def _read_lightcurve(self) :
        tel = 'CTIO'

        path_bool = []

        lc_dirc = os.path.join(config.LICVDIRC, self.field)
        for filt in ['B', 'V', 'I'] :
            lc_fname = f'{self.objname}_{tel}_{filt}.csv'
            lc_fpath = os.path.join(lc_dirc, lc_fname)
            lc_fpath_bool = os.path.isfile(lc_fpath)
            path_bool.append(lc_fpath_bool)

        if sum(path_bool) == 0 :
            self._read_from_sql()
            for filt in ['B', 'V', 'I'] :
                lc_fname = f'{self.objname}_{tel}_{filt}.csv'
                lc_fpath = os.path.join(lc_dirc, lc_fname)
                data = self.data_dict[filt]
                data.to_csv(lc_fpath)
        else :
            #_read_from_individual()
            for filt in ['B', 'V', 'I'] :
                lc_fname = f'{self.objname}_{tel}_{filt}.csv'
                lc_fpath = os.path.join(lc_dirc, lc_fname)
                self.data_dict[filt] = pd.read_csv(lc_fpath)


    def seeing_err_cut(self, data_dict) :
        filt_ls = ['B', 'V', 'I']
        result = dict.fromkeys(filt_ls)

        for filt in filt_ls :
            cat = data_dict[filt]
            
            seeing_cut_hi = 1.7
            seeing_cut_lo = 1.0
            bad_seeing = cat[(cat['seeing']*0.4 > seeing_cut_hi) | (cat['seeing']*0.4 < seeing_cut_lo)]
            good_seeing = cat[(cat['seeing']*0.4 <= seeing_cut_hi) & (cat['seeing']*0.4 >= seeing_cut_lo)]
        
            err_cut = np.median(good_seeing['magerr_inst']) + 3*np.std(good_seeing['magerr_inst'])
            good_err = good_seeing[good_seeing['magerr_inst'] < err_cut]
            bad_err = good_seeing[good_seeing['magerr_inst'] >= err_cut]

            result[filt] = good_err

        return result
    

    def run_drw_fit(self, filt, plot=True) :

        cat = self.data_dict[filt]
        cat_cln = self.seeing_err_cut(self.data_dict)[filt]

        cat_sorted = cat_cln.sort_values('mjd')

        x = np.array(cat_sorted['mjd']) * u.d
        y = np.array(cat_sorted['mag_inst']) * u.mag
        yerr = np.array(cat_sorted['magerr_inst']) * u.mag

        samples, gp, statuses = MCMC_fit(x, y, yerr, nwalkers=32, nburn=300, nsamp=2000, solver='minimize', suppress_warnings=True, jitter=True, clip=True)

        sig_vals = np.sqrt( np.exp(samples[:, 0])/2 )
        baseline = x[-1] - x[0]
        extra_t = int(baseline.value//10)
        t = np.linspace( x[0].value - extra_t, x[-1].value + extra_t, 1000 ).tolist()
        for i in range(len(x)):
            t.append(x[i].value)

        sort_ind = np.argsort(t)
        t = np.array(t)[sort_ind]

        mu, var = gp.predict(y.value, t, return_var=True)
        std = np.sqrt(var)

        tau_vals = 1/np.exp(samples[:, 1])
        sig_vals = np.sqrt( np.exp(samples[:, 0])/2 )
        jitter_vals = np.exp(samples[:, 2])
        sample_vals = np.vstack((np.log10(sig_vals), np.log10(tau_vals), np.log10(jitter_vals) )).T
        labels = [r'$\log_{10}\ (\sigma_{\rm DRW})$',
                r'$\log_{10}\ (\tau_{\rm DRW})$',
                r'$\log_{10}\ (\sigma_n)$']

        titles = []

        sig = np.log10(np.median(sig_vals))
        sig_err_l = sig - np.log10( np.percentile(sig_vals,16) )
        sig_err_u = np.log10( np.percentile(sig_vals,84) ) - sig
        sig_title = r'$' + '{:.2f}'.format(sig) + '^{' + ' +{:.2f}'.format(sig_err_u) +  '}_{' + '-{:.2f}'.format(sig_err_l) + '}$'
        titles.append(sig_title)

        tau = np.log10(np.median(tau_vals))
        tau_err_l = tau - np.log10( np.percentile(tau_vals,16) )
        tau_err_u = np.log10( np.percentile(tau_vals,84) ) - tau
        tau_title = r'$' + '{:.2f}'.format(tau) + '^{' + ' +{:.2f}'.format(tau_err_u) +  '}_{' + '-{:.2f}'.format(tau_err_l) + '}$'
        titles.append(tau_title)

        jit = np.log10(np.median(jitter_vals))
        jit_err_l = jit - np.log10( np.percentile(jitter_vals,16) )
        jit_err_u = np.log10( np.percentile(jitter_vals,84) ) - jit
        jit_title = r'$' + '{:.2f}'.format(jit) + '^{' + ' +{:.2f}'.format(jit_err_u) +  '}_{' + '-{:.2f}'.format(jit_err_l) + '}$'
        titles.append(jit_title)

        sigDRW_med = np.median(sig_vals)
        R_max_DRW = np.max(mu) - np.min(mu)
        std_dn = np.quantile(sig_vals, 0.25)
        std_up = np.quantile(sig_vals, 0.75)

        err = np.sqrt(var)
        err_top = mu + 1*err
        err_bot = mu - 1*err

        if plot :
            fig,ax = plt.subplots(2,1, figsize=(9,9), gridspec_kw={'height_ratios': [2, 1]})

            ax[0].scatter(
                x=cat['mjd'], y=cat['mag_inst'], s=5, c='gray', alpha=0.5, edgecolor='none'
            )


            ax[0].errorbar(
                x=cat_cln['mjd'], y=cat_cln['mag_inst'], yerr=cat_cln['magerr_inst'], fmt='o',
                c='k', alpha=0.3, mec='none',
                markersize=4, elinewidth=1, capsize=2, zorder=1,
                label='KAMP Observation'
            )

            ax[0].plot(t, mu, c='orange', zorder=2, label='DRW Fit')
            ax[0].fill_between(t, err_top, err_bot, color='orange', alpha=0.2, edgecolor='none', zorder=2, label=r'DRW Fit 1$\sigma$ error')

            ax[0].invert_yaxis()

            ax[0].set_title(self.objname)
            ax[0].set_xlabel('MJD [days]', fontsize=15)
            ax[0].set_ylabel('Magnitude [mag]', fontsize=15)
            ax[0].set_xlim(np.min(t), np.max(t))

            ax[0].legend(frameon=False)

            ax[1].hist(sig_vals, color='k', histtype=u'step', lw=2, bins=np.arange(0,0.35, 0.01))
            ax[1].hist(sig_vals, color='k', lw=2, alpha=0.3, bins=np.arange(0,0.35, 0.01))

            histmax = plt.ylim()[1]

            ax[1].fill_betweenx(np.linspace(0,histmax), std_dn, std_up, color='orange', alpha=0.2, edgecolor='none')
            ax[1].axvline(sigDRW_med, color='orange', ls='-', lw=2)

            ax[1].set_ylim(0,histmax)

            plot_text = rf"$\sigma_{{DRW}} = {{{sigDRW_med:.2f}}}^{{+{std_up-sigDRW_med:.2f}}}_{{-{sigDRW_med - std_dn:.1f}}}$ mag"
            ax[1].text(plt.xlim()[1]*0.5, plt.ylim()[1]*0.8, plot_text)

            ax[1].set_ylabel('Counts', fontsize=15)
            ax[1].set_xlabel(r'Posteriors of $\sigma_{DRW}$ [mag]', fontsize=15)

            plt.show()

        return t, mu, err, sig_vals
    
    def calc_sigDRW(self, filt) :
        t, mu, err, sig_vals = self.run_drw_fit(filt, plot=False)
        sigDRW_med = np.median(sig_vals)
        sigDRW_std = np.std(sig_vals)
        R_max_DRW = np.max(mu) - np.min(mu)
        return sigDRW_med, sigDRW_std, R_max_DRW
    
    def calc_fvar(self, filt):
        cat_cln = self.seeing_err_cut(self.data_dict)[filt]

        N = len(cat_cln)
        mag_med = np.median(cat_cln['mag_inst'])
        mse = np.median(cat_cln['magerr_inst']**2)

        summ = 0
        for i in range(N) : 
            summ += (cat_cln['mag_inst'].iloc[i] - mag_med)**2 - cat_cln['magerr_inst'].iloc[i]**2
        
        fvar = (1/mag_med) * np.sqrt(summ/N) if summ > 0 else np.nan
        fvar_sig = (np.sqrt(1/2/N) * mse/(mag_med**2*fvar))**2 + (np.sqrt(mse/N)/mag_med)**2 if fvar > 0 else np.nan

        return fvar, fvar_sig
    
    def calc_del_t(self, filt) :
        cat_cln = self.seeing_err_cut(self.data_dict)[filt]

        cat_sort = cat_cln.sort_values('mjd')
        del_t = [cat_sort['mjd'].iloc[i+1] - cat_sort['mjd'].iloc[i] for i in range(len(cat_sort)-1)]
        del_t_med = np.median(del_t)
        del_t_avg = np.mean(del_t)

        return del_t_med, del_t_avg

    def lc_prop_init(self) :
        dtype_lc_prop = np.dtype([
            ('agnname', str),
            ('filter', str),
            ('mse', float),
            ('N_obs', int),
            ('del_t_avg', float),
            ('del_t_med', float),
            ('mag_med', float),
            ('fvar', float),
            ('fvar_err', float),
            ('sig_DRW', float),
            ('sig_DRW_err', float),
            ('R_max', float),
            ('R_max_DRW', float)
        ])

        lc_prop = Table(dtype=dtype_lc_prop)
        return lc_prop


    
    def calc_lc_properties(self) :

        lc_prop_dir = os.path.join(config.LCPTDIRC, self.field)
        lc_prop_fname = f'{self.objname}.csv'
        lc_prop_fpath = os.path.join(lc_prop_dir, lc_prop_fname)

        if not os.path.isfile(lc_prop_fpath) : 

            lc_prop = self.lc_prop_init()
            filt_ls = ['B', 'V', 'I']
            for filt in filt_ls : 
                cat_cln = self.seeing_err_cut(self.data_dict)[filt]

                N = len(cat_cln)
                mag_med = np.median(cat_cln['mag_inst'])
                mse = np.median(cat_cln['magerr_inst']**2)
                R_max = np.max(cat_cln['mag_inst']) - np.min(cat_cln['mag_inst'])
                del_t_med, del_t_avg = self.calc_del_t(filt)
                fvar, fvar_sig = self.calc_fvar(filt)
                sigDRW_med, sigDRW_std, R_max_DRW = self.calc_sigDRW(filt)

                row = np.array([self.objname, filt, mse, N, del_t_avg, del_t_med, mag_med, fvar, fvar_sig, sigDRW_med, sigDRW_std, R_max, R_max_DRW])
                lc_prop.add_row(row)

            ascii.write(lc_prop, lc_prop_fpath)

        lc_prop = ascii.read(lc_prop_fpath)

        return lc_prop


        

    def cln_and_sort(self, table) :
        t = table.copy()
        t['nu'] = 3e8 / t['lmb']
        t['fnu'] = 10**((48.6 + t['mag'])/(-2.5))
        t['sed'] = t['nu'] * t['fnu']
        if 'errs' in t.keys() :
            t['ferr'] = 10**((48.6 + t['errs'])/(-2.5))
        
        t['mag'].fill_value = np.nan
        t_filled = t.filled()
        t_cln = t_filled[~np.isnan(t_filled['mag'])]
        t_cln.sort('nu')
        return t_cln


    def plot_lightcurve(self, plot_drw=False) -> None :
        """
        Plots light curve for object for BVI filters.
        """

        fig,ax = plt.subplots(3,1, figsize=(15,10))

        colors_ls = ['b', 'g', 'r']
        ecolors_ls = ['skyblue', 'yellowgreen', 'magenta']

        mjd_min = 10**10
        mjd_max = 10**10

        for pltidx in range(3) :

            filt = ['B', 'V', 'I'][pltidx]
            lc_data = self.data_dict[filt]
            lc_data_cut = self.seeing_err_cut(self.data_dict)[filt]

            
            if type(lc_data) != type(None) :
                
                lc_data_sort = lc_data_cut.sort_values('mjd')
                
                mjd_max = np.min([mjd_max, np.max(lc_data['mjd'])])
                mjd_min = np.min([mjd_min, np.min(lc_data['mjd'])])


                ax[pltidx].scatter(
                    x=lc_data['mjd'], y=lc_data['mag_inst'], edgecolor='none', c='gray', s=5, alpha=0.3,
                    zorder=0
                )

                ax[pltidx].errorbar(
                    x=lc_data_sort['mjd'], y=lc_data_sort['mag_inst'], yerr=lc_data_sort['magerr_inst'],
                    fmt='o',
                    mfc=colors_ls[pltidx], mec='none', ecolor=ecolors_ls[pltidx], elinewidth=1, capsize=2, markersize=5,
                    alpha=0.4,
                    zorder=1
                )

                if plot_drw :
                    t, mu, err, sig_vals = self.run_drw_fit(filt, plot=False)
                    err_top = mu + 1*err
                    err_bot = mu - 1*err

                    ax[pltidx].plot(t, mu, c='k', label='DRW Fit', zorder=3)
                    ax[pltidx].fill_between(t, err_top, err_bot, color='k', alpha=0.2, edgecolor='none', label=r'DRW Fit 1$\sigma$ error', zorder=4)

                
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
        ax[0].set_title(f'{self.objname}')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


    def plot_photSED(self) :

        def _annotate(start_range, end_range, ypos, label, trgt) :
            # Add an arrow
            trgt.annotate(
                '',
                xy=(end_range, ypos),
                xytext=(start_range, ypos),
                arrowprops=dict(arrowstyle='<->', lw=1),
            )
            
            # Add centered label
            trgt.text(
                (start_range + end_range) / 2,
                ypos*0.99,  # Slightly below the arrow
                label,
                fontsize=12,
                ha='center',
                va='center', backgroundcolor='none'
            )

        mag_dict = {
            'Gmag' : 6397.4,
            'BPmag': 5164.7,
            'RPmag': 7830.5, #
            'FUV'  : 1538.6,
            'XNUV' : 2315.7, #
            'umag' : 3543,
            'gmag' : 4770,
            'rmag' : 6231,
            'imag' : 7625,
            'zmag' : 9134, #
            'Ymag' : 10200,
            'Jmag' : 12200,
            'Hmag' : 16300,
            'Kmag' : 21900, #
            'W1mag': 34000,
            'W2mag': 46000,
            'W3mag': 120000,
            'W4mag': 220000
        }

        agncat = AgnCatalogs(self.field)
        agncat.match_cat('lqac6')
        agncat.match_cat('gaiaqsocand')
        sed_data = agncat.catalog[np.where(agncat.catalog['Jname'] == self.objname)][0]
        z = agncat.catalog[np.where(agncat.catalog['Jname'] == self.objname)][0]['gaiaqsocand_z']

        gaia_keys = ['Gmag', 'BPmag', 'RPmag']
        uv_keys = ['FUV', 'XNUV']
        opt_keys = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
        nir_keys = ['Ymag', 'Jmag', 'Hmag', 'Kmag']
        wise_keys = ['W1mag', 'W2mag', 'W3mag', 'W4mag']

        keys_ls = [gaia_keys, uv_keys, opt_keys, nir_keys, wise_keys]
        keys_nm = ['Gaia', 'UV', 'Optical', 'NIR', 'WISE']
        c_ls = ['green', 'purple', 'blue', 'magenta', 'red']

        # external catalogs
        lmbs = [mag_dict[mag]*1e-10 for mag in list(mag_dict.keys())]
        mags = [sed_data[f'lqac6_{mag}'] for mag in list(mag_dict.keys())]
        sed_table = Table(
            data = [list(mag_dict.keys()), lmbs, mags],
            names= ['band', 'lmb', 'mag']
        )

        # prekamp
        data_cut = self.seeing_err_cut(self.data_dict)
        band_p = ['B', 'V', 'I']
        lmbs_p = np.array([4450, 5510, 8060])*1e-10
        mags_p = [np.median(data_cut[filt]['mag_inst']) for filt in band_p]
        errs_p = [np.std(data_cut[filt]['magerr_inst']) for filt in band_p]

        sed_table_pkamp = Table(
            data = [band_p, lmbs_p, mags_p, errs_p],
            names= ['band', 'lmb', 'mag', 'errs']
        )

        pkamp_sed = self.cln_and_sort(sed_table_pkamp)
        extcat_sed= self.cln_and_sort(sed_table)

        fig,ax = plt.subplots(1,2,figsize=(14,5))

        ax[0].scatter(np.log10(extcat_sed['nu']), np.log10(extcat_sed['sed']), c='gray', s=20, edgecolor='gray')
        ax[0].plot(np.log10(extcat_sed['nu']), np.log10(extcat_sed['sed']), c='gray', ls='--')

        data_cut = self.seeing_err_cut(self.data_dict)

        ax[0].errorbar(
            x=np.log10(pkamp_sed['nu']), y=np.log10(pkamp_sed['sed']), yerr=pkamp_sed['errs'], 
            c='k', fmt='*', elinewidth=1, capsize=2, label='preKAMP',  markersize=8
        )

        plt_max = np.max(extcat_sed['sed'])
        plt_max = np.log10(plt_max)+0.5
        ant_max = plt_max - 0.25

        _annotate(np.log10(7.5e14),17,ant_max, 'UV', ax[0])
        _annotate(np.log10(4e14),np.log10(7.5e14),ant_max, 'vis.', ax[0])
        _annotate(np.log10(1e14),np.log10(4e14),ant_max, 'NIR', ax[0])
        _annotate(np.log10(1e13),np.log10(1e14),ant_max, 'IR', ax[0])

        ax[0].set_xlim(13,17)
        btm, _ = ax[0].set_ylim()
        ax[0].set_ylim(btm, plt_max)

        ax[0].grid(alpha=0.3)

        opt_keys = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']
        gaia_keys = ['Gmag', 'BPmag', 'RPmag']

        extcat_sed_opt = extcat_sed[np.where(np.isin(extcat_sed['band'], opt_keys))]
        extcat_sed_gaia = extcat_sed[np.where(np.isin(extcat_sed['band'], gaia_keys))]

        ax[1].scatter(np.log10(extcat_sed_opt['nu']), np.log10(extcat_sed_opt['sed']), c='gray', s=30, edgecolor='gray', label='external cat.')
        ax[1].scatter(np.log10(extcat_sed_gaia['nu']), np.log10(extcat_sed_gaia['sed']), c='green', s=30, edgecolor='green', marker='^', label='Gaia')
        ax[1].plot(np.log10(extcat_sed_gaia['nu']), np.log10(extcat_sed_gaia['sed']), c='green', ls=':')
        ax[1].plot(np.log10(extcat_sed_opt['nu']), np.log10(extcat_sed_opt['sed']), c='gray', ls='--')
        ax[1].errorbar(
            x=np.log10(pkamp_sed['nu']), y=np.log10(pkamp_sed['sed']), yerr=pkamp_sed['errs'], 
            c='k', fmt='*', elinewidth=1, capsize=2, label='preKAMP',  markersize=16
        )
        ax[1].legend()

        xl, xu = ax[1].set_xlim()
        yl, yu = ax[1].set_ylim()

        rect = plt.Rectangle((xl, yl), xu - xl, yu - yl, fill=False, color='blue', linewidth=1, alpha=0.5)
        ax[0].add_patch(rect)

        fig.supxlabel(r'$\log \nu$ [Hz]')
        ax[0].set_ylabel(r'relative  $\log\nu f_{\nu}$ [erg s$^{-1}$cm$^{-2}$]')
        fig.suptitle(fr'{self.objname} ($z_{{quaia}}$ = {z:.2f})')

        plt.show()