"""
Methods for detecting and combining files
"""

from __future__ import annotations

import os

import numpy as np

__all__: list[str] = [
    "files_in_dir",
    "combine_images",
    "combine_videos",
    "combine_pdfs",
]


def select_filenames(filenames: list[str], suf: str = "", pref: str = "") -> list[str]:
    return [f for f in filenames if (f.endswith(suf) and f.startswith(pref))]


def files_in_dir(
    dir: str,
    sort: bool = True,
    include_subdirs: bool = False,
    suf: str = "",
    pref: str = "",
) -> list[str]:
    """
    Select files from directory matching filename conditions.

    Scans a directory for files matching optional prefix and suffix filters.
    Can optionally include subdirectories and sort results.

    Args:
        dir: Absolute path to directory to search
        sort: If True, sort filenames alphabetically (default: True)
        include_subdirs: If True, search subdirectories recursively (default: False)
        suf: Required suffix to include file, e.g., '.txt' (default: '')
        pref: Required prefix to include file (default: '')

    Returns:
        List of absolute file paths matching the criteria

    Example:
        >>> files = files_in_dir('/path/to/dir', suf='.py')
        >>> files = files_in_dir('/path/to/dir', pref='test_', suf='.txt', include_subdirs=True)
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
    files: list[str] | None = None,
    file_dir: str = ".",
    save_as: str = "combined_image.pdf",
    save_to: str | None = None,
    size: tuple[int, int] = (1000, 1000),
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Merge multiple image files into a single combined image file.

    Creates a grid layout of images and saves as PDF or image. Images are
    thumbnailed to fit the grid based on total count. Uses PIL for processing.

    Args:
        files: List of image file paths. If None, scans file_dir
        file_dir: Directory to search for images if files not provided (default: '.')
        save_as: Output filename (default: 'combined_image.pdf')
        save_to: Directory to save output. If None, uses file_dir
        size: Output image dimensions in pixels (default: (1000, 1000))
        figsize: Override thumbnail size as (width, height) pixels

    Example:
        >>> combine_images(files=['img1.jpg', 'img2.jpg'], save_as='grid.pdf')
        >>> combine_images(file_dir='/path/to/images', size=(2000, 2000))
    """

    def get_dxy(N: int, size: tuple[int, int] = (1000, 1000)) -> tuple[int, int]:
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


def combine_videos(
    files: list[str] | None = None,
    file_dir: str = ".",
    save_to: str | None = None,
    save_as: str = "combined_videos.mp4",
) -> None:
    """
    Merge multiple video files into a single side-by-side video.

    Uses ffmpeg to horizontally stack videos. All videos must have identical
    duration. Requires ffmpeg and ffprobe installed on system.

    Args:
        files: List of video file paths (.mp4). If None, scans file_dir for .mp4 files
        file_dir: Directory to search for videos if files not provided (default: '.')
        save_to: Directory to save output. If None, uses file_dir
        save_as: Output filename (default: 'combined_videos.mp4')

    Raises:
        ValueError: If fewer than 2 videos provided or durations don't match

    Example:
        >>> combine_videos(files=['vid1.mp4', 'vid2.mp4'], save_as='stacked.mp4')
        >>> combine_videos(file_dir='/path/to/videos')
    """
    if files is None:
        files = files_in_dir(file_dir, suf=".mp4")

    if len(files) < 2:
        raise ValueError("At least two video files are required to combine.")

    # Check if all videos have the same duration
    durations = []
    for file in files:
        result = (
            os.popen(
                f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file}"'
            )
            .read()
            .strip()
        )
        durations.append(float(result))

    if len(set(durations)) != 1:
        raise ValueError("All videos must have the same duration.")

    if save_to is None:
        save_to = file_dir
    filepath = os.path.join(save_to, save_as)

    filter_complex = ""
    for i in range(len(files)):
        filter_complex += f"[{i}:v]"

    filter_complex += f"hstack=inputs={len(files)}[outv]"

    input_files = " ".join([f"-i {file}" for file in files])
    os.system(
        f'ffmpeg {input_files} -filter_complex "{filter_complex}" -map "[outv]" -c:v libx264 -crf 23 {filepath}'
    )

    print(f"Videos combined as {filepath}")


def combine_pdfs(
    files: list[str] | None = None,
    file_dir: str = ".",
    pref: str = "",
    save_to: str | None = None,
    save_as: str = "final.pdf",
    include_subdirs: bool = True,
) -> None:
    """
    Merge multiple PDF files into a single combined PDF.

    Concatenates PDFs in order, preserving all pages. Uses pypdf library
    for merging. Can filter by filename prefix.

    Args:
        files: List of PDF file paths. If None, scans file_dir for .pdf files
        file_dir: Directory to search for PDFs if files not provided (default: '.')
        pref: Required filename prefix to include (default: '' for all)
        save_to: Directory to save output. If None, uses file_dir
        save_as: Output filename (default: 'final.pdf')
        include_subdirs: If True, search subdirectories recursively (default: True)

    Example:
        >>> combine_pdfs(files=['doc1.pdf', 'doc2.pdf'], save_as='merged.pdf')
        >>> combine_pdfs(file_dir='/reports', pref='2024_', save_as='all_2024.pdf')
    """
    if files is None:
        files = files_in_dir(
            file_dir, include_subdirs=include_subdirs, pref=pref, suf=".pdf"
        )
    import pypdf

    # Use PdfWriter instead of deprecated PdfMerger (removed in pypdf 5.0.0)
    writer = pypdf.PdfWriter()
    for f in files:
        reader = pypdf.PdfReader(open(f, "rb"))
        for page in reader.pages:
            writer.add_page(page)

    if save_to is None:
        save_to = file_dir
    filepath = os.path.join(save_to, save_as)
    with open(filepath, "wb") as output_file:
        writer.write(output_file)
    print(f"Concatenated pdfs saved as {filepath}")
