"""
Methods for importing data in lab-specific formats
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Callable, List

import os

from ..process.import_aux import (
    constrain_selected_tracks,
    finalize_timeseries_dataframe,
    generate_dataframes,
    get_Schleyer_metadata_inv_x,
    init_endpoint_dataframe_from_timeseries,
    match_larva_ids,
    read_timeseries_from_raw_files_per_larva,
    read_timeseries_from_raw_files_per_parameter,
)

__all__: list[str] = [
    "import_Jovanic",
    "import_Schleyer",
    "import_Berni",
    "import_Arguello",
    "lab_specific_import_functions",
]


def import_Jovanic(
    source_id: str,
    source_dir: str,
    tracker: Any,
    filesystem: Any,
    match_ids: bool = True,
    matchID_kws: Dict[str, Any] = {},
    interpolate_ticks: bool = True,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
     Builds a larvaworld dataset from Jovanic-lab-specific raw data

    Parameters
    ----------
     source_id : string
         The ID of the imported dataset
     source_dir : string
         The folder containing the imported dataset
     match_ids : boolean
         Whether to use the match-ID algorithm
         Defaults to True
     matchID_kws : dict
         Additional keyword arguments to be passed to the match-ID algorithm.
     interpolate_ticks : boolean
         Whether to interpolate timeseries into a fixed timestep timeseries
         Defaults to True
    **kwargs: keyword arguments
         Additional keyword arguments to be passed to the constrain_selected_tracks function.


    Returns
    -------
     s : pandas.DataFrame
         The timeseries dataframe
     e : pandas.DataFrame
         The endpoint dataframe

    """
    s0 = read_timeseries_from_raw_files_per_parameter(
        pref=f"{source_dir}/{source_id}", tracker=tracker
    )

    if match_ids:
        s0 = match_larva_ids(s0, Npoints=tracker.Npoints, dt=tracker.dt, **matchID_kws)

    s0 = constrain_selected_tracks(s0, **kwargs)

    e = init_endpoint_dataframe_from_timeseries(df=s0, dt=tracker.dt)
    s = finalize_timeseries_dataframe(
        s0, complete_ticks=False, interpolate_ticks=interpolate_ticks
    )
    return s, e


def import_Schleyer(
    source_dir: str,
    tracker: Any,
    filesystem: Any,
    save_mode: str = "semifull",
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
     Builds a larvaworld dataset from Schleyer-lab-specific raw data.
     The data is available at https://doi.gin.g-node.org/10.12751/g-node.5e1ifd/

    Parameters
    ----------
     source_dir : string
         The folder containing the imported dataset
     save_mode : string
         Mode to define the sequence of columns/parameters to store.
         Defaults to 'semi-full'
    **kwargs: keyword arguments
         Additional keyword arguments to be passed to the generate_dataframes function.


    Returns
    -------
     s : pandas.DataFrame
         The timeseries dataframe
     e : pandas.DataFrame
         The endpoint dataframe

    """
    if type(source_dir) == str:
        source_dir = [source_dir]

    dfs = []
    for f in source_dir:
        inv_x = get_Schleyer_metadata_inv_x(dir=f)
        files = [os.path.join(f, n) for n in os.listdir(f) if n.endswith(".csv")]
        dfs += read_timeseries_from_raw_files_per_larva(
            files=files,
            inv_x=inv_x,
            read_sequence=filesystem.read_sequence,
            save_mode=save_mode,
            tracker=tracker,
        )

    return generate_dataframes(dfs, tracker.dt, **kwargs)


def import_Berni(
    source_files: List[str], tracker: Any, filesystem: Any, **kwargs: Any
) -> Tuple[Any, Any]:
    """
     Builds a larvaworld dataset from Berni-lab-specific raw data

    Parameters
    ----------
     source_files : list
         List of the absolute filepaths of the data files.
    **kwargs: keyword arguments
         Additional keyword arguments to be passed to the generate_dataframes function.


    Returns
    -------
     s : pandas.DataFrame
         The timeseries dataframe
     e : pandas.DataFrame
         The endpoint dataframe

    """
    dfs = read_timeseries_from_raw_files_per_larva(
        files=source_files, read_sequence=filesystem.read_sequence, tracker=tracker
    )
    return generate_dataframes(dfs, tracker.dt, **kwargs)


def import_Arguello(
    source_files: List[str], tracker: Any, filesystem: Any, **kwargs: Any
) -> Tuple[Any, Any]:
    """
     Builds a larvaworld dataset from Arguello-lab-specific raw data

    Parameters
    ----------
     source_files : list
         List of the absolute filepaths of the data files.
    **kwargs: keyword arguments
         Additional keyword arguments to be passed to the generate_dataframes function.


    Returns
    -------
     s : pandas.DataFrame
         The timeseries dataframe
     e : pandas.DataFrame
         The endpoint dataframe

    """
    dfs = read_timeseries_from_raw_files_per_larva(
        files=source_files, read_sequence=filesystem.read_sequence, tracker=tracker
    )
    return generate_dataframes(dfs, tracker.dt, **kwargs)


lab_specific_import_functions: dict[str, Callable[..., tuple[Any, Any]]] = {
    "Jovanic": import_Jovanic,
    "Berni": import_Berni,
    "Schleyer": import_Schleyer,
    "Arguello": import_Arguello,
}
