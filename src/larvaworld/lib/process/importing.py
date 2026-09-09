"""
Methods for importing data in lab-specific formats
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Callable, List

import glob
import os

from ... import vprint
from ..process.import_aux import (
    PER_PARAMETER_TXT_SUFFIXES,
    constrain_selected_tracks,
    convert_spine_files_to_per_parameter_txt,
    finalize_timeseries_dataframe,
    generate_dataframes,
    get_Schleyer_metadata_inv_x,
    init_endpoint_dataframe_from_timeseries,
    match_larva_ids,
    read_deeplabcut_tracks,
    read_timeseries_from_raw_files_per_larva,
    read_timeseries_from_raw_files_per_parameter,
)

__all__: list[str] = [
    "import_Jovanic",
    "import_Schleyer",
    "import_Berni",
    "import_Arguello",
    "import_DeepLabCut",
    "lab_specific_import_functions",
]


def _ensure_Jovanic_per_parameter_files(
    source_id: str, source_dir: str, Npoints: int
) -> None:
    """
    Converts raw tracker spine files of a dataset into the per-parameter txt files.

    The Jovanic format reads one txt file per recorded quantity, while the raw tracker
    emits one `.spine` file per recording session. If the txt files are missing but spine
    files are present, they are generated here so that a dataset can be imported straight
    from the raw tracker output. Datasets already stored as txt files are left untouched.

    Args:
        source_id: The ID of the dataset, used as the filename prefix.
        source_dir: The folder holding the dataset's files.
        Npoints: The number of tracked midline points.

    """
    if all(
        os.path.isfile(f"{source_dir}/{source_id}_{suf}.txt")
        for suf in PER_PARAMETER_TXT_SUFFIXES
    ):
        return
    # Spine files either sit in a folder named after the dataset, one subfolder per
    # recording session, or directly next to the other datasets of the experiment.
    spine_files = sorted(
        glob.glob(f"{source_dir}/{source_id}/**/*.spine", recursive=True)
    ) or sorted(glob.glob(f"{source_dir}/{source_id}*.spine"))
    if not spine_files:
        return
    vprint(
        f"**--- Found {len(spine_files)} raw spine files for '{source_id}'. "
        f"Converting to the per-parameter format -----",
        1,
    )
    convert_spine_files_to_per_parameter_txt(
        source_files=spine_files,
        target_dir=source_dir,
        source_id=source_id,
        Npoints=Npoints,
        # Prefixing with the dataset ID keeps the agent IDs of datasets that are imported
        # together and compared in one plot distinct from each other.
        id_prefix=f"{source_id}_",
    )


def import_Jovanic(
    source_id: str,
    source_dir: str,
    tracker: Any,
    filesystem: Any,
    match_ids: bool = True,
    matchID_kws: Dict[str, Any] = {},
    interpolate_ticks: bool = True,
    estimate_dt: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
     Builds a larvaworld dataset from Jovanic-lab-specific raw data

     The data is read from one txt file per recorded quantity. If those are missing but
     raw tracker `.spine` files are present, they are converted first, so that a dataset
     can be imported directly from the raw tracker output.

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
     estimate_dt : boolean
         Whether to estimate the tracker timestep from the timestamps of the data
         instead of using the lab-format's nominal value.
         Defaults to False
    **kwargs: keyword arguments
         Additional keyword arguments to be passed to the constrain_selected_tracks function.


    Returns
    -------
     s : pandas.DataFrame
         The timeseries dataframe
     e : pandas.DataFrame
         The endpoint dataframe

    """
    _ensure_Jovanic_per_parameter_files(source_id, source_dir, tracker.Npoints)

    s0 = read_timeseries_from_raw_files_per_parameter(
        pref=f"{source_dir}/{source_id}", tracker=tracker, estimate_dt=estimate_dt
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


def import_DeepLabCut(
    source_dir: str | list[str],
    tracker: Any,
    filesystem: Any,
    parent_dir: str = ".",
    merged: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Build a Larvaworld dataset from single-animal DeepLabCut exports."""
    dfs, npoints = read_deeplabcut_tracks(
        source_dir=source_dir,
        parent_dir=parent_dir,
        merged=merged,
        pixel_to_mm=filesystem.pixel_to_mm,
    )
    tracker.Npoints = npoints
    return generate_dataframes(dfs, tracker.dt, **kwargs)


lab_specific_import_functions: dict[str, Callable[..., tuple[Any, Any]]] = {
    "Jovanic": import_Jovanic,
    "Berni": import_Berni,
    "Schleyer": import_Schleyer,
    "Arguello": import_Arguello,
    "DeepLabCut": import_DeepLabCut,
}
