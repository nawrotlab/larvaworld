
from lib.aux import dictsNlists as dNl

from lib.plot import dataplot as dplt,traj as traj, stridecycle as cycle, table as tab, grid as grid, epochs as epochs
# from lib.anal.fitting import test_boutGens

graph_dict = dNl.NestDict({
    'crawl pars': dplt.plot_crawl_pars,
    'angular pars': dplt.plot_ang_pars,
    'endpoint params': dplt.plot_endpoint_params,
    'powerspectrum': dplt.powerspectrum,
    'stride Dbend': dplt.plot_stride_Dbend,
    'stride Dor': dplt.plot_stride_Dorient,
    'stride cycle': cycle.stride_cycle,
    'interference': dplt.plot_interference,
    'dispersal': dplt.plot_dispersion,
    'dispersal summary': grid.dsp_summary,
    'runs & pauses': dplt.plot_stridesNpauses,
    'epochs': epochs.plot_bouts,
    'fft': epochs.plot_fft_multi,
    'turn duration': dplt.plot_turn_duration,
    'turn amplitude': dplt.plot_turns,
    'stride track': traj.annotated_strideplot,
    'turn track': traj.annotated_turnplot,
    'marked strides': traj.plot_marked_strides,
    'sample tracks': traj.plot_sample_tracks,
    'trajectories': traj.traj_grouped,
    'turn amplitude VS Y pos': dplt.plot_turn_amp,
    'turn Dbearing to center': dplt.plot_turn_Dorient2center,
    'chunk Dbearing to source': dplt.plot_chunk_Dorient2source,
    'C odor (real)': dplt.plot_odor_concentration,
    'C odor (perceived)': dplt.plot_sensed_odor_concentration,
    'navigation index': dplt.plot_navigation_index,
    'Y pos': dplt.plot_Y_pos,
    'PI (boxplot)': dplt.boxplot_PI,
    'pathlength': dplt.plot_pathlength,
    'food intake (timeplot)': dplt.plot_food_amount,
    'gut': dplt.plot_gut,
    'food intake (barplot)': dplt.intake_barplot,
    'deb': dplt.plot_debs,
    'timeplot': dplt.timeplot,
    'ethogram': dplt.plot_ethogram,
    'foraging': dplt.plot_foraging,
    'barplot': dplt.barplot,
    'scatter': dplt.plot_2pars,
    'nengo': dplt.plot_nengo_network,
    'ggboxplot': dplt.ggboxplot
})


ModelGraphDict = dNl.NestDict({
    'configuration': tab.modelConfTable,
    'sample track': grid.test_model,
    # 'sample epochs': test_boutGens,
    'module hists': dplt.module_endpoint_hists,
    'summary': grid.model_summary,

})


if __name__ == "__main__":
    # from lib.plot.grid import test_model
    # from lib.plot.grid import model_summary
    # from lib.anal.fitting import test_boutGens
    # print('dddddddddddd')
    # test_boutGens(refID='None.150controls', mID='explorer', show=True)
    _=ModelGraphDict['configuration'](mID='PHIonNEU', show=True)
    # test_model(mID='explorer', dur=2.1 / 3, dt=1 / 16, Nids=1, min_turn_amp=20, show=True)

