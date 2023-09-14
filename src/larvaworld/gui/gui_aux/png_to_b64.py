import PySimpleGUI as sg
import os
import base64
from PIL import Image
from io import BytesIO


__all__ = [
    'png_to_b64',
    'bs64_to_png'
]

def png_to_b64():
    # Base64 Encoder - encodes a folder of PNG files and creates a .py file with definitions

    '''
        Make base64 images
        input:  folder with .png .ico .gif 'sigma
        output: output.py file with variables
    '''
    OUTPUT_FILENAME = 'output.py'

    folder = sg.popup_get_folder('Source folder for images\nImages will be encoded and results saved to %s'%OUTPUT_FILENAME,
                               title='Base64 Encoder')

    if not folder:
        sg.popup_cancel('Cancelled - No valid folder entered')
        return
    try:
        namesonly = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.ico') or f.endswith('.gif')]
    except:
        sg.popup_cancel('Cancelled - No valid folder entered')
        return

    outfile = open(os.path.join(f'{folder}/..', OUTPUT_FILENAME), 'w')

    for i, file in enumerate(namesonly):
        contents = open(os.path.join(folder, file), 'rb').read()
        encoded = base64.b64encode(contents)
        file=file.replace('-', '_')
        outfile.write('\n{} = {}'.format(file[:file.index(".")], encoded))
        sg.OneLineProgressMeter('Base64 Encoding', i+1, len(namesonly), key='-METER-')

    outfile.close()
    sg.popup('Completed!', 'Encoded %s files'%(i+1))

def bs64_to_png(bs64, save_as=None, save_to='.'):


    k = Image.open(BytesIO(base64.b64decode(bs64)))

    if save_as is not None:
        f = f'{save_to}/{save_as}.png'
        k.save(f, 'PNG')
    return k

if __name__ == '__main__':
    png_to_b64()