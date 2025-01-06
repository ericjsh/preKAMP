from astropy.io import ascii
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import match_coordinates_sky, SkyCoord
from astroquery.vizier import Vizier

import matplotlib.pyplot as plt

from astropy.table.table import Table

import numpy as np
import pandas as pd
import os

import dask.dataframe as dd

from lc_tools.lightcurve_config import *

# TODO: depreciate data reading method for kmtn_catalog. This should be doen by pkamp_lightcurve.LightCurve. Focus on catalog matching.


class kmtn_catalog :

    def __init__(self, INPUTVAR) :
        self.INPUTVAR = INPUTVAR
        self.field, self.year, self.tel, self.filt = INPUTVAR

        kmtn_cat_fname = f'matched_targets_{self.field}.csv'
        kmtn_cat_fpath = os.path.join(DATAPATH, kmtn_cat_fname)

        self.simple_catalog = ascii.read(kmtn_cat_fpath)
        self.cent_ra = np.median(self.simple_catalog['ra'])
        self.cent_dec = np.median(self.simple_catalog['dec'])

        self.generate_star_catalog()

    def read_data(self) -> None :

        df_fname = '_'.join(self.INPUTVAR)+'.csv'
        df_fpath = os.path.join(DATAPATH, df_fname)

        #self.lightcurve_catalog = pd.read_csv(df_fpath, sep=' ')

        ddf = dd.read_csv(df_fpath, sep=' ')
        self.lightcurve_catalog = ddf.compute()



    def _load_gaia_cat(self) -> Table :
        '''
        Calling GAIA DR3 via Vizier as astropy.table.table.Table format.
        Catalogue cut at gaia_cat_cut illustrated at:
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        Catalouge should be named as 'gaia_{field}.csv'
        '''
        gaia_cat_fname = f'gaia_{self.field}.csv'

        if gaia_cat_fname in os.listdir(DATAPATH) :
            gaia_cat_path = os.path.join(DATAPATH, gaia_cat_fname)
            gaia_cat = ascii.read(gaia_cat_path)

        else : 
            print(f'{gaia_cat_fname} not in local directory. Downloading from Vizier ...')
            Vizier.ROW_LIMIT = -1

            gaia = Vizier.query_region(
                coord.SkyCoord(
                    ra=self.cent_ra, dec=self.cent_dec, 
                    unit=(u.deg, u.deg), frame='icrs'
                ),
                radius=2*u.deg,
                catalog=['Gaia'],
            )

            gaia_cat = gaia['I/356/qsocand']

            output_fpath = os.path.join(DATAPATH, gaia_cat_fname)
            ascii.write(gaia_cat, output_fpath, overwrite=True)
        
        self.quaia_cat = gaia_cat
        #return gaia_cat



    def _match_kmtn_gaia(self, plot=False) -> Table :
        '''
        Cross matching sources from preKAMP NGC55 and GAIA DR3
        '''

        self._load_gaia_cat()

        kmtn_radec = SkyCoord(
            ra=self.simple_catalog['ra']*u.deg,
            dec=self.simple_catalog['dec']*u.deg,
        )

        gaia_cat_radec = SkyCoord(
            ra=self.quaia_cat['RA_ICRS']*u.deg,
            dec=self.quaia_cat['DE_ICRS']*u.deg,
            frame='icrs'
        )

        mcat = self.simple_catalog.copy()

        idx, d2d, d3d = kmtn_radec.match_to_catalog_sky(gaia_cat_radec)

        mcat['sep'] = d2d.to(u.arcsec)

        keys = ["Source", "SolID", "Class", "PQSO", "PGal", "z"]
        key_dict = {key: "quaia_"+key for key in keys}
        for key in key_dict :
            mcat[key_dict[key]] = self.quaia_cat[idx][key]

        self.kmtn_quaia_cat = mcat[np.where(mcat['sep'] < 2*u.arcsec)]

        if plot :
            plt.figure(figsize=(10,10))

            plt.scatter(
                x=self.simple_catalog['ra'], y=self.simple_catalog['dec'], s=1, c='gray', alpha=0.5
            )

            plt.scatter(
                x=self.kmtn_quaia_cat['ra'], y=self.kmtn_quaia_cat['dec'], s=20, c=self.kmtn_quaia_cat['quaia_PQSO'], cmap='Reds'
            )

            plt.colorbar(label='pQSO')

            plt.gca().invert_xaxis()

            plt.xlabel('RA [deg]')
            plt.ylabel('DEC [deg]')

            plt.show()

    def _match_kmtn_mqs(self) :

        mqs_cat = Table.read('/Users/kamp/Desktop/KAMP/pkamp/pkamp-lightcurve/data/milliquas.fits')

        kmtn_radec = SkyCoord(
            ra=self.simple_catalog['ra']*u.deg,
            dec=self.simple_catalog['dec']*u.deg,
        )

        mqs_radec = SkyCoord(
            ra=mqs_cat['RA']*u.deg,
            dec=mqs_cat['DEC']*u.deg
        )

        mcat = self.simple_catalog.copy()

        idx, d2d, d3d = kmtn_radec.match_to_catalog_sky(mqs_radec)

        mcat['sep'] = d2d.to(u.arcsec)
        mcat['B_mqs'] = mqs_cat[idx]['BMAG']

        self.kmtn_mqs_cat = mcat[np.where(mcat['sep'] < 2*u.arcsec)]

    def generate_lightcurve(self, objname) :
        light_curve = self.lightcurve_catalog[
            (self.lightcurve_catalog['Jname' ]== objname)
        ]

        return light_curve

    def generate_star_catalog(self) :

        star_cat_fname = f'starcand_{self.field}.csv'
        star_cat_fpath = os.path.join(DATAPATH, star_cat_fname)
        star_cat = ascii.read(star_cat_fpath)

        self.star_cat = star_cat
