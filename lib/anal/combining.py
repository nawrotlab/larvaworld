import os
import six
from PyPDF2 import PdfFileMerger, PdfFileReader
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def combine_images(filenames=None, file_dir='.', save_as='combined_image.pdf', save_to='.', size=(1000, 1000),
                   figsize=None):
    if filenames is None:
        filenames = [f for f in os.listdir(f'{file_dir}/') if os.path.isfile(os.path.join(f'{file_dir}/', f))]
    filenames = np.sort(filenames)
    files = [os.path.join(file_dir, name) for name in filenames]
    x, y = size
    if figsize is None:
        if len(files) <= 4:
            dx = int(x / 2)
        elif len(files) <= 9:
            dx = int(x / 3)
        else:
            dx = int(x / 4)
        dy = int(dx * y / x)
    else:
        dx, dy = figsize
    # print(filenames)
    new_im = Image.new('RGB', size)

    index = 0
    for j in np.arange(0, y, dy):
        for i in np.arange(0, x, dx):
            im = Image.open(files[index])
            im.thumbnail((dx, dy))
            new_im.paste(im, (i, j))
            index += 1
            # print(index)
            if index > len(files) - 1:
                break
        else:
            continue
        break

    file_path = os.path.join(save_to, save_as)
    new_im.save(file_path)
    print(f'Images combined as {file_path}')


def combine_videos_4to1(filenames=None, file_dir='.', save_as='combined_videos.mp4', save_to='.'):
    if filenames is None:
        filenames = [f for f in os.listdir(f'{file_dir}/') if os.path.isfile(os.path.join(f'{file_dir}/', f))]
    filenames = np.sort(filenames)
    files = [os.path.join(file_dir, name) for name in filenames]
    temp_files = [os.path.join(file_dir, name) for name in ['output1.mp4', 'output2.mp4']]
    final_file = os.path.join(save_to, save_as)
    os.system(
        f'ffmpeg -i {files[0]} -i {files[1]} -filter_complex "[0:v]pad=iw*2:ih[int]; [int][1:v]overlay=W/2:0[vid]" -map "[vid]" -c:v libx264 -crf 23  {temp_files[0]}')
    os.system(
        f'ffmpeg -i {files[2]} -i {files[3]} -filter_complex "[0:v]pad=iw*2:ih[int]; [int][1:v]overlay=W/2:0[vid]" -map "[vid]" -c:v libx264 -crf 23  {temp_files[1]}')
    os.system(
        f'ffmpeg -i {temp_files[0]} -i {temp_files[1]} -filter_complex "[0:v]pad=iw*2:ih[int]; [int][1:v]overlay=W/2:0[vid]" -map "[vid]" -c:v libx264 -crf 23  {final_file}')

    print(f'Videos combined as {final_file}')

def append_pdf(input, output):
    [output.addPage(input.getPage(page_num)) for page_num in range(input.numPages)]

def combine_pdfs(file_dir='.', save_as="final.pdf", pref='') :
    merger = PdfFileMerger()
    files=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for filename in [f for f in filenames if (f.endswith(".pdf") and f.startswith(pref))]:
            files.append(os.path.join(dirpath, filename))
    # print(files)
    files.sort()
    for f in files :
        merger.append(PdfFileReader(open(f, 'rb')))
    filepath = os.path.join(file_dir, save_as)
    merger.write(filepath)
    print(f'Concatenated pdfs saved as {filepath}')


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'W'], edge_color='W',
                     bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    else :
        fig  = None

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='W')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax, fig