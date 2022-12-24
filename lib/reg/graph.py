import os

from lib.aux import dictsNlists as dNl
from lib import reg
from lib import plot


class GraphRegistry:
    def __init__(self):
        self.dict = reg.funcs.graphs
        # self.graphgroups=dNl.NestDict()

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
        ds = {}
        for graphgroup in graphgroups:
            if graphgroup in self.graphgroups.keys():
                entries = self.graphgroups[graphgroup]
                dir= f'{save_to}/{graphgroup}' if save_to is not None else None
                ds[graphgroup] = self.eval(entries, save_to=dir,**kws)
        return ds


    def entry(self, ID, name=None, args={}):
        assert self.get(ID)
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
        from lib.conf.stored.analysis_conf import analysis_dict
        return analysis_dict

    def model_tables(self, mIDs,dIDs=None, save_to=None, **kwargs):
        from lib.aux.combining import combine_pdfs
        ds = {}
        ds['mdiff_table'] = self.dict['model diff'](mIDs,dIDs=dIDs, save_to=save_to, **kwargs)
        gfunc=self.dict['model table']
        for mID in mIDs:
            try:
                ds[f'{mID}_table'] = gfunc(mID, save_to=save_to, **kwargs)
            except:
                print('TABLE FAIL', mID)
        if save_to is not None and len(ds)>1 :
            combine_pdfs(file_dir=save_to, save_as="_MODEL_TABLES_.pdf", deep=False)
        return dNl.NestDict(ds)

    def model_summaries(self, mIDs, save_to=None, **kwargs):
        from lib.aux.combining import combine_pdfs
        ds = {}
        for mID in mIDs:
            try:
                ds[f'{mID}_summary'] = self.dict['model summary'](mID, save_to=save_to, **kwargs)
            except:
                print('SUMMARY FAIL', mID)
        if save_to is not None and len(ds)>0 :
            combine_pdfs(file_dir=save_to, save_as="_MODEL_SUMMARIES_.pdf", deep=False)
        return ds

    def store_model_graphs(self, mIDs, dir):
        from lib import reg
        f1 = reg.datapath('model_tables', dir)
        f2 = reg.datapath('model_summaries', dir)
        os.makedirs(f1, exist_ok=True)
        os.makedirs(f2, exist_ok=True)

        graphs = dNl.NestDict({
            'tables': self.model_tables(mIDs, save_to=f1),
            'summaries': self.model_summaries(mIDs, Nids=10, refDataset=self, save_to=f2)
        })
        return graphs


graphs = GraphRegistry()
