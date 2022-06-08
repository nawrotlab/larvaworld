from typing import Union

import numpy as np
from matplotlib import cm
from scipy.stats import ks_2samp
import pandas as pd

import lib.aux.dictsNlists
import lib.aux.naming as nam

from lib.conf.base.pars import getPar, ParDict

class ExpFitter:
    from lib.stor.larva_dataset import LarvaDataset
    def __init__(self, sample: Union[dict, str, LarvaDataset], stat_coefs=None,
                 valid_fields=['angular motion', 'reorientation', 'spatial motion', 'dispersion curve',
                               'stride cycle curve'],
                 use_symbols=False, overwrite=False):
        from lib.stor.larva_dataset import LarvaDataset
        if isinstance(sample, LarvaDataset):
            self.sample = sample
            self.sample_conf = self.sample.config
        else:
            if type(sample) == dict:
                self.sample_conf = sample
            elif type(sample) == str:
                from lib.conf.stored.conf import loadConf
                self.sample_conf = loadConf(sample, 'Ref')
            self.sample = LarvaDataset(self.sample_conf['dir'], load_data=True)

        key = 's' if use_symbols else 'd'
        self.valid_fields = valid_fields
        self.df = self.multicol_df(key)
        columns = lib.aux.dictsNlists.flatten_list(self.df['cols'].values.tolist())
        self.m_col = pd.MultiIndex.from_tuples(columns, names=['Field', 'Pars'])

        if overwrite:
            self.build(use_symbols, stat_coefs)

        elif not self.retrieve():
            self.build(use_symbols, stat_coefs)

    def retrieve(self):
        dic = self.sample.load_ExpFitter()
        if dic is not None:
            print('Loaded existing configuration')
            self.sample_data = {k: np.array(v) for k, v in dic['sample_data'].items()}
            self.stat_coefs = dic['stat_coefs']
            # temp=np.array()
            self.df_st = pd.DataFrame(dic['stats'], columns=self.m_col)
            # self.df=pd.DataFrame.from_dict(dic['df'])
            # columns = fun.flatten_list(self.df['cols'].values.tolist())
            # self.m_col = pd.MultiIndex.from_tuples(columns, names=['Field', 'Pars'])
            # self.df_st = pd.DataFrame(columns=self.m_col)
            return True
        else:
            return False

    def build(self, use_symbols, stat_coefs):
        self.sample_data = self.get_data(self.sample, self.df)
        if stat_coefs is None:
            stat_coefs = self.default_stat_coefs
        self.stat_coefs = stat_coefs
        self.df_st = pd.DataFrame(columns=self.m_col)
        self.store()
        print('New configuration built and stored')

    @property
    def default_stat_coefs(self):
        N=5/len(self.valid_fields)
        coefs = {}
        for field, pars in self.df['pars'].items():
            if field in ['angular motion', 'reorientation', 'spatial motion']:
                for p in pars:
                    coefs[p] = N / len(pars)
            elif field in ['dispersion curve']:
                for p in pars:
                    coefs[p] = N / len(pars)
            elif field in ['stride cycle curve']:
                for p in pars:
                    coefs[p] = N / len(pars)
        return coefs

    def store(self):
        sample_data_lists = {k: v.tolist() for k, v in self.sample_data.items()}
        # temp={'sample_data': sample_data_lists, 'df': self.df.to_dict(),
        temp = {'sample_data': sample_data_lists, 'stat_coefs': self.stat_coefs,
                'stats': self.df_st.values.tolist()}

        self.sample.save_ExpFitter(temp, )

    def multicol_df(self, key='s'):
        shorts = [
            ['fov', 'foa', 'b', 'bv', 'ba'],
            ['str_N', 'str_tr', 'cum_d', 'v_mu'],
            # ['str_N', 'str_tr', 'cum_d', 'v_mu', 'tor5_mu'],
            ['str_fo', 'str_b', 'tur_fo', 'tur_t'],
            ['dsp', 'sdsp'],
            ['sv', 'fov', 'bv'],
        ]
        groups = [
            'angular motion',
            'spatial motion',
            'reorientation',
            'dispersion curve',
            'stride cycle curve',
        ]
        group_cols = ['Oranges', 'Purples', 'Greens', 'Reds', 'Blues']
        df = pd.DataFrame(
            {'group': groups,
             'shorts': shorts,
             'group_color': group_cols
             })

        df['pars'] = df['shorts'].apply(lambda row: getPar(row))
        df['symbols'] = df['shorts'].apply(lambda row: getPar(row, to_return=key))
        df['cols'] = df.apply(lambda row: [(row['group'], p) for p in row['symbols']], axis=1)
        df['par_colors'] = df.apply(
            lambda row: [cm.get_cmap(row['group_color'])(i) for i in np.linspace(0.4, 0.7, len(row['pars']))], axis=1)
        df.set_index('group', inplace=True)
        return df

    def simple_df(self):
        pass

    def get_data(self, d, df):
        s, e = d.step_data, d.endpoint_data
        d_d = {}
        if 'angular motion' in self.valid_fields:
            d_d.update({p: np.abs(d.get_par(p).dropna().values.flatten()) for p in df['pars'].loc['angular motion']})
        if 'spatial motion' in self.valid_fields:
            d_d.update({p: e[p].values for p in df['pars'].loc['spatial motion']})
        if 'reorientation' in self.valid_fields:
            d_d.update({p: d.get_par(p).dropna().values.flatten() for p in df['pars'].loc['reorientation']})

        if 'dispersion curve' in self.valid_fields:
            t0, t1 = int(0 * d.fr), int(40 * d.fr)
            d_d['dispersion'] = d.load_aux('dispersion', 'dispersion')['median'][t0:t1]
            d_d['scaled_dispersion'] = d.load_aux('dispersion', nam.scal('dispersion'))['median'][t0:t1]

        if 'stride cycle curve' in self.valid_fields:
            for p in df['pars'].loc['stride cycle curve']:
                chunk_d = d.load_aux('stride', p).values
                if any([x in p for x in ['bend', 'orientation']]):
                    chunk_d = np.abs(chunk_d)
                d_d[f'str_{p}'] = np.nanquantile(chunk_d, q=0.5, axis=0)
        return d_d

    def compare(self, d, save_to_config=False):
        df = self.df
        ref_d = self.sample_data
        d_d = self.get_data(d, df)
        idx = d.id

        df_st0 = pd.DataFrame(index=[idx], columns=self.m_col)
        # df_pv0 = pd.DataFrame(index=[idx],columns=m_col)
        for g in df.index:
            ps, cs = df[['pars', 'cols']].loc[g].values
            for p, c in zip(ps, cs):
                if g in ['angular motion', 'reorientation', 'spatial motion']:
                    if g in self.valid_fields:
                        if d_d[p].shape[0] != 0:
                            st, pv = ks_2samp(ref_d[p], d_d[p])
                        else:
                            st = 1.0
                        df_st0[c].loc[idx] = st
                        # df_pv0[c].loc[idx] = pv
                elif g in ['dispersion curve']:
                    if g in self.valid_fields:
                        dist = np.sqrt(np.sum((ref_d[p] - d_d[p]) ** 2)) / np.sum(np.abs(ref_d[p]))
                        df_st0[c].loc[idx] = dist
                elif g in ['stride cycle curve']:
                    if g in self.valid_fields:
                        dist = np.sqrt(np.sum((ref_d[f'str_{p}'] - d_d[f'str_{p}']) ** 2)) / np.sum(
                            np.abs(ref_d[f'str_{p}']))
                        df_st0[c].loc[idx] = dist
        self.df_st = self.df_st.append(df_st0)
        df_st00 = df_st0.droplevel('Field', axis=1).loc[idx]
        if save_to_config:
            dic = df_st00.to_dict()
            d.config['sample_fit'] = dic
            d.save_config()
        fit = self.get_fit(df_st00)
        return fit

    def compare_short(self, d):
        df = self.df
        ref_d = self.sample_data
        d_d = self.get_data(d, df)
        stats = {}
        for g in df.index:
            ps, cs = df[['pars', 'cols']].loc[g].values
            for p, c in zip(ps, cs):
                if g in ['angular motion', 'reorientation', 'spatial motion']:
                    st, pv = ks_2samp(ref_d[p], d_d[p])
                    stats[c] = st
                elif g in ['dispersion curve']:
                    dist = np.sqrt(np.sum((ref_d[p] - d_d[p]) ** 2)) / np.sum(np.abs(ref_d[p]))
                    stats[c] = dist
                elif g in ['stride cycle curve']:
                    dist = np.sqrt(np.sum((ref_d[f'str_{p}'] - d_d[f'str_{p}']) ** 2)) / np.sum(
                        np.abs(ref_d[f'str_{p}']))
                    stats[c] = dist
        print(stats)
        fit = self.get_fit(stats)
        return fit

    def get_fit(self, stats):
        cs = self.stat_coefs
        fit = np.nansum([s * cs[p] for p, s in stats.items()])
        return fit




if __name__ == '__main__':
    ref_dir = '/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/processed/FRUvsQUI/FRU->PUR/AM/test_10l'
    d_dir = '/home/panos/nawrot_larvaworld/larvaworld/data/SimGroup/single_runs/imitation/test_10l_imitation_0'
    from lib.stor.larva_dataset import LarvaDataset

    ref = LarvaDataset(ref_dir)
    c = ref.config

    f = ExpFitter(c)
    # print(f.df['pars'])
    d = LarvaDataset(d_dir)
    fit = f.compare(d)
    print(fit)
