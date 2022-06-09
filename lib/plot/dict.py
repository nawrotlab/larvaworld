from lib.plot.plot_datasets import plot_odor_concentration, plot_sensed_odor_concentration, plot_Y_pos, boxplot_PI, \
    plot_pathlength, plot_food_amount, intake_barplot, plot_gut, plot_debs, timeplot, plot_ethogram, plot_foraging, \
    barplot, plot_2pars, plot_nengo_network, ggboxplot, plot_turns, plot_ang_pars, plot_crawl_pars, plot_turn_duration, \
    plot_turn_amp, plot_stride_Dbend, plot_marked_strides, plot_sample_tracks, plot_stride_Dorient, plot_interference, \
    plot_dispersion, powerspectrum, plot_navigation_index, plot_stridesNpauses, plot_endpoint_params, \
    plot_chunk_Dorient2source, plot_turn_Dorient2center
from lib.plot.plotting import plot_trajectories

graph_dict = {
    'crawl pars': plot_crawl_pars,
    'angular pars': plot_ang_pars,
    'endpoint params': plot_endpoint_params,
    'powerspectrum': powerspectrum,
    'stride Dbend': plot_stride_Dbend,
    'stride Dor': plot_stride_Dorient,
    'interference': plot_interference,
    'dispersion': plot_dispersion,
    'runs & pauses': plot_stridesNpauses,
    'turn duration': plot_turn_duration,
    'turn amplitude': plot_turns,
    'marked strides': plot_marked_strides,
    'sample tracks': plot_sample_tracks,
    'trajectories': plot_trajectories,
    'turn amplitude VS Y pos': plot_turn_amp,
    'turn Dbearing to center': plot_turn_Dorient2center,
    'chunk Dbearing to source': plot_chunk_Dorient2source,
    'C odor (real)': plot_odor_concentration,
    'C odor (perceived)': plot_sensed_odor_concentration,
    'navigation index': plot_navigation_index,
    'Y pos': plot_Y_pos,
    'PI (boxplot)': boxplot_PI,
    'pathlength': plot_pathlength,
    'food intake (timeplot)': plot_food_amount,
    'gut': plot_gut,
    'food intake (barplot)': intake_barplot,
    'deb': plot_debs,
    'timeplot': timeplot,
    'ethogram': plot_ethogram,
    'foraging': plot_foraging,
    'barplot': barplot,
    'scatter': plot_2pars,
    'nengo': plot_nengo_network,
    'ggboxplot': ggboxplot
}


