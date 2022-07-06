from lib.aux import dictsNlists as dNl


def build_mod_dict():
    from lib.plot import table, hist, grid
    d = dNl.NestDict({
        'configuration': table.modelConfTable,
        'sample track': grid.test_model,
        # 'sample epochs': test_boutGens,
        'module hists': hist.module_endpoint_hists,
        'summary': grid.model_summary,
    })
    return d


def build():
    from lib.plot import bar, bearing, box, deb, epochs, freq, grid, hist, stridecycle, time, traj, table
    d = dNl.NestDict()
    d['table'] = dNl.NestDict({
        'mpl': table.mpl_table,
        # 'barplot': bar.barplot,
        # 'auto_barplot': bar.auto_barplot,
    })

    d['bar'] = dNl.NestDict({
        'food intake (barplot)': bar.intake_barplot,
        'barplot': bar.barplot,
        'auto_barplot': bar.auto_barplot,
    })
    d['bearing'] = dNl.NestDict({
        'bearing/turn': bearing.plot_turn_Dbearing,
        'bearing to center/turn': bearing.plot_turn_Dorient2center,
        'bearing to source/epoch': bearing.plot_chunk_Dorient2source,

    })
    d['box'] = dNl.NestDict({
        'lineplot': box.lineplot,
        'PI (combo)': box.boxplot_PI,
        'PI (simple)': box.PIboxplot,
        'boxplot (grouped)': box.boxplot,
        'boxplot (simple)': box.boxplots,
        'foraging': box.plot_foraging,
        'ggboxplot': box.ggboxplot,
        'double patch': box.boxplot_double_patch,
    })
    d['deb'] = dNl.NestDict({
        'food intake (timeplot)': deb.plot_food_amount,
        'gut': deb.plot_gut,
        'deb': deb.plot_debs,
    })
    d['epochs'] = dNl.NestDict({
        'runs & pauses': epochs.plot_stridesNpauses,
        'epochs': epochs.plot_bouts,
    })
    d['freq'] = dNl.NestDict({
        'powerspectrum': freq.powerspectrum,
        'fft': freq.plot_fft_multi,

    })
    d['grid'] = dNl.NestDict({
        'double-patch summary': grid.DoublePatch_summary,
        'RvsS summary': grid.RvsS_summary,
        'dispersal summary': grid.dsp_summary,
        'chemotaxis summary': grid.chemo_summary,
        'eval summary': grid.result_summary
    })
    d['hist'] = dNl.NestDict({
        'crawl pars': hist.plot_crawl_pars,
        'angular pars': hist.plot_ang_pars,
        'angular/epoch': hist.plot_bout_ang_pars,
        'endpoint pars (hist)': hist.plot_endpoint_params,
        'endpoint pars (scatter)': hist.plot_endpoint_scatter,
        'turn duration': hist.plot_turn_duration,
        'turn amplitude': hist.plot_turns,
        'turn amplitude VS Y pos': hist.plot_turn_amp_VS_Ypos,
    })
    d['stridecycle'] = dNl.NestDict({
        'stride Dbend': stridecycle.plot_stride_Dbend,
        'stride Dor': stridecycle.plot_stride_Dorient,
        'stride cycle': stridecycle.stride_cycle,
        'interference': stridecycle.plot_interference,
    })
    d['time'] = dNl.NestDict({
        'dispersal': time.plot_dispersion,
        'C odor (real)': time.plot_odor_concentration,
        'C odor (perceived)': time.plot_sensed_odor_concentration,
        'navigation index': time.plot_navigation_index,
        'Y pos': time.plot_Y_pos,
        'pathlength': time.plot_pathlength,
        'timeplot': time.timeplot,
        'autoplot': time.auto_timeplot,
        'ethogram': time.plot_ethogram,
        'nengo': time.plot_nengo_network,
    })
    d['traj'] = dNl.NestDict({
        'stride track': traj.annotated_strideplot,
        'turn track': traj.annotated_turnplot,
        'marked strides': traj.plot_marked_strides,
        'sample tracks': traj.plot_sample_tracks,
        'trajectories': traj.traj_grouped
    })
    return d


def build_error_dict():
    from lib.plot import table, bar, grid
    d = dNl.NestDict({
        'error table': table.error_table,
        'error summary': grid.eval_summary,
        'error barplot': bar.error_barplot,
        # 'sample epochs': test_boutGens,
        # 'module hists': hist.module_endpoint_hists,
        # 'summary': grid.model_summary,
    })
    return d


class GraphDict:
    def __init__(self):
        self.grouped_dic = build()
        self.flat_dict = dNl.flatten_dict(self.grouped_dic)
        self.dict = dNl.NestDict(dNl.merge_dicts([dic for k, dic in self.grouped_dic.items()]))
        self.mod_dict = build_mod_dict()
        self.error_dict = build_error_dict()

    def get(self, f):
        if isinstance(f, str):
            if f in self.dict.keys():
                f = self.dict[f]
            elif f in self.mod_dict.keys():
                f = self.mod_dict[f]
            elif f in self.flat_dict.keys():
                f = self.flat_dict[f]
            elif f in self.error_dict.keys():
                f = self.error_dict[f]
            else:
                raise
        return f

    def eval0(self, entry, **kws):
        func = self.get(entry['plotID'])
        d = {entry['title']: func(**entry['args'], **kws)}
        return d

    def eval(self, entries, **kws):
        ds = {}
        for entry in entries:

            d = self.eval0(entry, **kws)
            ds.update(d)
        return ds

    def entry(self, ID, title=None, args={}):
        assert self.get(ID)
        if title is None:
            title = ID
        return {'title': title, 'plotID': ID, 'args': args}


graph_dict = GraphDict()

if __name__ == '__main__':

    # print(graph_dict.get('endpoint plot'))
    # print(graph_dict.get('endpoint plot'))
    # f='fff'
    # assert graph_dict.get('turn amplitude VS Y pos')
    print('DDDDD')
    # assert graph_dict.get(f)
    # print('DDDfffDD')