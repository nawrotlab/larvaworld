import os

from larvaworld.lib.aux import naming as nam
from larvaworld.lib import reg, aux


class GraphRegistry:
    def __init__(self):
        self.dict = reg.funcs.graphs

    @property
    def ks(self):
        return list(self.dict.keys())

    def get(self, f):
        if isinstance(f, str):
            if f in self.dict.keys():
                f = self.dict[f]
            else:
                raise
        return f

    def eval0(self, entry, **kws):
        func = self.get(entry['plotID'])
        d = {entry['key']: func(**entry['args'], **kws)}
        return d



    def eval(self, entries, **kws):

        ds = {}
        for entry in entries:

            d = self.eval0(entry, **kws)
            ds.update(d)
        return ds

    def eval_graphgroups(self, graphgroups,save_to=None,**kws):
        kws.update({'subfolder' : None})
        ds = aux.AttrDict()
        for gg in graphgroups:
            if isinstance(gg, dict):
                for ggID, entries in gg.items() :
                    dir = f'{save_to}/{ggID}' if save_to is not None else None
                    ds[ggID] = self.eval(entries, save_to=dir, **kws)
            elif isinstance(gg, str) and gg in self.graphgroups.keys():
                ggID=gg
                entries = self.graphgroups[ggID]
                dir = f'{save_to}/{ggID}' if save_to is not None else None
                ds[ggID] = self.eval(entries, save_to=dir, **kws)
            else :
                raise

        return ds

    def run(self, ID, **kwargs):
        func = self.get(ID)
        return func(**kwargs)

    def entry(self, ID, name=None, **kwargs):
        assert self.get(ID)
        args=kwargs
        if name is not None:
            args['name']=name
            key=name
        else :
            key = ID
        return {'key': key, 'plotID': ID, 'args': args}

    def group_entry(self, gID, entrylist):
        self.graphgroups[gID]=entrylist

    @property
    def graphgroups(self):
        from larvaworld.lib.reg.stored.analysis_conf import analysis_dict
        return analysis_dict

    def model_tables(self, mIDs,dIDs=None, save_to=None, **kwargs):
        ds = {}
        ds['mdiff_table'] = self.dict['model diff'](mIDs,dIDs=dIDs, save_to=save_to, **kwargs)
        gfunc=self.dict['model table']
        for mID in mIDs:
            try:
                ds[f'{mID}_table'] = gfunc(mID, save_to=save_to, **kwargs)
            except:
                print('TABLE FAIL', mID)
        if save_to is not None and len(ds)>1 :
            aux.combine_pdfs(file_dir=save_to, save_as="_MODEL_TABLES_.pdf", deep=False)
        return aux.AttrDict(ds)

    def model_summaries(self, mIDs, save_to=None, **kwargs):
        ds = {}
        for mID in mIDs:
            try:
                ds[f'{mID}_summary'] = self.dict['model summary'](mID, save_to=save_to, **kwargs)
            except:
                print('SUMMARY FAIL', mID)
        if save_to is not None and len(ds)>0 :
            aux.combine_pdfs(file_dir=save_to, save_as="_MODEL_SUMMARIES_.pdf", deep=False)
        return ds

    def store_model_graphs(self, mIDs, dir):
        f1 = reg.datapath('model_tables', dir)
        f2 = reg.datapath('model_summaries', dir)
        os.makedirs(f1, exist_ok=True)
        os.makedirs(f2, exist_ok=True)

        graphs = aux.AttrDict({
            'tables': self.model_tables(mIDs, save_to=f1),
            'summaries': self.model_summaries(mIDs, Nids=10, refDataset=self, save_to=f2)
        })
        return graphs

    def source_graphgroup(self, source_ID, pos=None, **kwargs):
        gID = f"locomotion relative to source {source_ID}"
        d0 = []
        for ref_angle, name in zip([None, 270], [f'bearing to {source_ID}', 'bearing to 270deg']):
            entry=self.entry('bearing/turn', name=name, **{"min_angle":5.0, "ref_angle":ref_angle, "source_ID":source_ID, **kwargs} )
            d0.append(entry)

        for p in [nam.bearing2(source_ID), nam.dst2(source_ID), nam.scal(nam.dst2(source_ID))] :
            d0.append(self.entry('timeplot', name=p, **{"pars":[p], **kwargs}))

        for chunk in ['stride', 'pause', 'Lturn', 'Rturn']:
            for dur in [0.0, 0.5, 1.0]:
                name = f'{chunk}_bearing2_{source_ID}_min_{dur}_sec'
                d0.append(
                    self.entry('bearing to source/epoch', name=name, **{
                        "min_dur" : dur, "chunk" : chunk, "source_ID":source_ID, **kwargs}))
        return aux.AttrDict({gID: d0})

    def multisource_graphgroup(self, sources, **kwargs):
        graphgroups = []
        for source_ID, pos in sources.items():
            graphgroups.append(self.source_graphgroup(source_ID, pos=pos, **kwargs))
        return graphgroups

    def get_analysis_graphgroups(self, exp, sources, **kwargs):
        groups = ["traj", "general"]
        groups += self.multisource_graphgroup(sources, **kwargs)

        if exp in ['random_food']:
            groups.append("survival")
        else:
            dic = {
                'patch': ["patch"],
                'tactile': ["tactile"],
                'thermo': ["thermo"],
                'RvsS': ["deb", "intake"],
                'growth': ["deb", "intake"],
                'anemo': ["anemotaxis"],
                'puff': ["puff"],
                'chemo': ["chemo"],
                'RL': ["RL"],
                # 'dispersion': ['comparative_analysis'],
                'dispersion': ["endpoint", "distro", "dsp"],

            }
            for k, v in dic.items():
                if k in exp:
                    groups += v

        return groups



graphs = GraphRegistry()

