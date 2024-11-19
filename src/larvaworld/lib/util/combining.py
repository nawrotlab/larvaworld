"""
Methods for detecting and combining files
"""

import os

import numpy as np

__all__ = [
    "files_in_dir",
    "combine_images",
    "combine_videos_4to1",
    "combine_pdfs",
]


def select_filenames(filenames, suf="", pref=""):
    return [f for f in filenames if (f.endswith(suf) and f.startswith(pref))]


def files_in_dir(dir, sort=True, include_subdirs=False, suf="", pref=""):
    """
    Select files from directory fulfilling filename conditions

    Parameters
    ----------
    - dir: string
        Absolute path to folder where to look for files.
    - pref: string
        Required prefix to include file (default: '').
    - suf: string
        Required suffix to include file (default: '').
    - sort: bool
        Sort filenames (default: True).
    - include_subdirs: bool
        Include files in subdirectories (default: False).

    Returns
    -------
    - list of strings
        A list of the filenames selected.

    """
    fs = []
    if include_subdirs:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in select_filenames(filenames, suf=suf, pref=pref):
                fs.append(os.path.join(dirpath, filename))
    else:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in select_filenames(filenames, suf=suf, pref=pref):
                fs.append(os.path.join(dirpath, filename))
            break
    if sort:
        fs = np.sort(fs)
    return fs


def combine_images(
    files=None,
    file_dir=".",
    save_as="combined_image.pdf",
    save_to=None,
    size=(1000, 1000),
    figsize=None,
):
    """
    Merge multiple image files into one single file and store it

    Parameters
    ----------
    - files: list of strings
        List of image absolute filenames (optional).
    - file_dir: string
        Absolute path to folder where to look for files.
    - save_to: string
        Absolute path to folder where to store file (optional).
    - save_as: string
        Filename to store pdf as (default: 'combined_image.pdf').
    - size : tuple
        Image size(default: (1000, 1000))
    - figsize : tuple
        Figure size (optional)

    """

    def get_dxy(N, size=(1000, 1000)):
        x, y = size
        if N <= 4:
            dx = int(x / 2)
        elif N <= 9:
            dx = int(x / 3)
        else:
            dx = int(x / 4)
        dy = int(dx * y / x)
        return dx, dy

    from PIL import Image

    if files is None:
        files = files_in_dir(file_dir)

    if figsize is None:
        N = len(files)
        dx, dy = get_dxy(N, size=size)
    else:
        dx, dy = figsize

    x, y = size
    new_im = Image.new("RGB", size)

    index = 0
    for j in np.arange(0, y, dy):
        for i in np.arange(0, x, dx):
            im = Image.open(files[index])
            im.thumbnail((dx, dy))
            new_im.paste(im, (i, j))
            index += 1
            if index > len(files) - 1:
                break
        else:
            continue
        break

    if save_to is None:
        save_to = file_dir
    filepath = os.path.join(save_to, save_as)
    new_im.save(filepath)
    print(f"Images combined as {filepath}")


def combine_videos_4to1(
    files=None, file_dir=".", save_to=None, save_as="combined_videos.mp4"
):
    """
    Merge 4 video files into one single file and store it

    Parameters
    ----------
    - files: list of strings
        List of image absolute filenames (optional).
    - file_dir: string
        Absolute path to folder where to look for files.
    - save_to: string
        Absolute path to folder where to store file (optional).
    - save_as: string
        Filename to store pdf as (default: 'combined_videos.mp4').

    """
    if files is None:
        files = files_in_dir(file_dir)
    temp_files = [os.path.join(file_dir, n) for n in ["output1.mp4", "output2.mp4"]]

    if save_to is None:
        save_to = file_dir
    filepath = os.path.join(save_to, save_as)

    os.system(
        f'ffmpeg -i {files[0]} -i {files[1]} -filter_complex "[0:v]pad=iw*2:ih[int]; [int][1:v]overlay=W/2:0[vid]" -map "[vid]" -c:v libx264 -crf 23  {temp_files[0]}'
    )
    os.system(
        f'ffmpeg -i {files[2]} -i {files[3]} -filter_complex "[0:v]pad=iw*2:ih[int]; [int][1:v]overlay=W/2:0[vid]" -map "[vid]" -c:v libx264 -crf 23  {temp_files[1]}'
    )
    os.system(
        f'ffmpeg -i {temp_files[0]} -i {temp_files[1]} -filter_complex "[0:v]pad=iw*2:ih[int]; [int][1:v]overlay=W/2:0[vid]" -map "[vid]" -c:v libx264 -crf 23  {filepath}'
    )

    print(f"Videos combined as {filepath}")


def combine_pdfs(
    files=None,
    file_dir=".",
    pref="",
    save_to=None,
    save_as="final.pdf",
    include_subdirs=True,
):
    """
    Merge multiple pdf files into one single pdf and store it

    Parameters
    ----------
    - files: list of strings
        List of pdf absolute filenames (optional).
    - file_dir: string
        Absolute path to folder where to look for files.
    - pref: string
        Required prefix to include file (default: '').
    - save_to: string
        Absolute path to folder where to store file (optional).
    - save_as: string
        Filename to store pdf as (default: False).
    - include_subdirs: bool
        Include files in subdirectories (default: True).

    """
    if files is None:
        files = files_in_dir(
            file_dir, include_subdirs=include_subdirs, pref=pref, suf=".pdf"
        )
    import pypdf

    merger = pypdf.PdfMerger()
    for f in files:
        merger.append(pypdf.PdfReader(open(f, "rb")))

    if save_to is None:
        save_to = file_dir
    filepath = os.path.join(save_to, save_as)
    merger.write(filepath)
    print(f"Concatenated pdfs saved as {filepath}")
