import copy
import inspect
import os
from ast import literal_eval
from typing import List, Tuple, Type

import numpy as np
import PySimpleGUI as sg
import operator

from matplotlib import ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from lib.conf import par_conf
from lib.conf.conf import loadConfDict, saveConf, deleteConf
import lib.aux.functions as fun
from lib.stor.paths import get_parent_dir
import lib.conf.dtype_dicts as dtypes

on_image = b'iVBORw0KGgoAAAANSUhEUgAAAFoAAAAnCAYAAACPFF8dAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAHGElEQVRo3u2b3W8T6RWHnzMzSbDj4KTkq1GAfFCSFrENatnQikpFC2oqRWhXq92uKm7aKy5ou9cV1/wFvQAJqTdV260qaLdSF6RsS5tN+WiRFopwTRISNuCAyRIF8jHJeObtxYyd8diYhNjBEI70KvZ4rGie9ze/c877joVAtLW19ezcuXPvpk2bIgAKxYsMQbifnDRvjcW13d1v1DY2NIm1ZM1RhmGa5tzw8PC/x8fHrymlnOzr8KKjo+NbR48e/VV3d/e+yWSC+fm5AohVnlfFD0c5/O3SJ0QjX+GdQ+8TqY4QiUTQNK3sICulsCyL+fl5RkdHr506depYLBb7LAt0T0/PD44fP3720ueDoTMDv2P6yUNEVFBay2BlndTsCD95+2e89d0+urq62LZtG4ZhUM4xOztLLBZjYmLCPHHixLtXr179K4Bs3ry54eTJk/HzQx/XfXzh97kQ04DFB3gdQIsN+3sOcfSDD+nt7WXLli0A2LaNbdtlB1jXdXRdz7y/fv068Xh87tixY7uTyeSY0d/f//OpmYd1f7nwUV7ISgAtG3IW9JIoGSSl8fZbP6K9vT0DOX17WpZVdqArKyvRNA0RF8yuXbtIJpPVhw8f/vD06dO/MHp7ew9/9p9PUQGrUGm43l//e5VP2UUELyY017fSVN/M1q1bl4+LUFVVRWVlZdmBFpEM5LTCW1pa2LNnzyEAo6mpqW3yy0SuXaShaoDu/dV8xyihlZjQWPdVAMLhcMELKueIRCK0trZ+Xdd1wwiHw5sdx862Cy0A2QClB4BLniRZpNA00ETjZY+0IJRS5KTwjP+KD7IBeLD9ys6cX+x4+RnnhJHXAjxVpxXtV7XSfRZSqjv4lQWdr4XxeXQasDIC9lGiUk/JRgDtT4bis4m0inWfmv2TUkyTlg2iaL9PK5+NpEu8nNr6FYVTMtD+W1bl6wbzjdexBuso0Iz44aswqK2gqgELtCTIg+y1J6fNVb82AaR8C0bbvbx3Z6ODfkbY3wC7N7tCsAHtPuifgiy6oO39oKpAvwH6leUJSH0PRIE2vjHujOcqpJxWsL/jAtOvQMVZMM6BJMFpBvtAnonZBapu43r66kErsHu8fv6Kq1SZBi0BFefc9tlpAVWfa0Wp/RvXo7Xn+YZqdMFptwOfpUC766m+yXfccr1bNYDT/Rr0ysLrFHE8Hw4K1/ReVGWr2Rj0vHkvqNCrAU8p9dSx9mRoe0N3k1wQdgbiUmACZkC/DvY3wd4HL3IrMh+IYp8T3G5bPWgHZMq1D6cT9Ju+zyrcRAluqRf0dv1zcDrcgcqdjGJcuIg889z1AB1cyl09aAH9GqQOgb3X8+q7QAhS33YtQ+67FUi+u0EfglTf6qoOx3HWBU4xJ2HtisatffXLYL/p1tJ2r28eHoLx9wLfTbhJ1OlYnZodxykbiCv5P/79w8KgVf7XotzuUL8B2pjX4UXcikOSoN0LqP9ybruuXwJt0vP6FSr6ZQMdPCcLtKhlpgIo5YOsfMN7L3OgxwrbjDaS26CICRJfeePyLNDlYhn+zwuCzgBULmRJg3W8kT7ueCt5an06vLWCLgd/L2wdahkwjnurp5eepZSQ1co8upySX/CcFSmaoJJtkPT6tA9yqZ7vCD4k9TRFl6NlFAbt92FZBi0e5Axgr45O77BIqdaknWcrer3soFiTZeRTU8aHxX00K0vt3paW+B8VKzFoEckCXc6WUbCOzupifLaR5cfKU7dG1g6LUHxVu5O9fAGVlZUsLCy8cDtY6Tm6rlNRUZH1uWFZFvXRRvKWec5ymZdJfnkenilFMpx+MoVSsLi4SCgUoqKiAtM0n7poUw52kX6Kqq6uDhFhYWEh85ygce/evZneN/ZH/3H13DI45dvYdjzIDrl7hSUs7SYejPNkboZEIkFnZyfRaBQR4fHjxywuLq4I1vMAXstEhEIhGhoaCIVCKKWYnJwkmUwuKKWUMTQ0dPHIkSN9+3Z/n0v/vZAN219deGBlnXa+HVJ88s8/U1e7hebmZqqrq4lGo9TU1KyoS3wRISIZbx4dHWV2dpaLFy9eVkrZ+uzs7Nz27ds/6DvQz5JpMX53FCfQG4uncFG+0kuVeACjX8TpbO0itehQU1NDOBxG07SyHrZtE4/HGR4eJh6Pc+bMmV9OT0/fMO7cufOngYGBs5ZlvfNe3xH6D7zL/8ZusrAw9xTFrt+vWhzH4Y/nf8uDqfuYpkkkEiEajZblTysAlpaWePToEaZpEovFGBwcHBgbG/soc/MbhhE5ePDgH9rb23/Y0tJCbW0thmG4PlQGm6g3R24w9eVDvta2k8b6JnS9vH5eIbhJ0LIsZmbcvHL79u3zAwMD76VSqSdZLisismPHjh93dXX9tLGx8U3DMCK8jtUm28VEIvGvW7du/XpkZOQ3ypcx/w+op8ZtEbCnywAAAABJRU5ErkJggg=='

off_image = b'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'

off_image_disabled = b'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'
on_image_disabled = b'iVBORw0KGgoAAAANSUhEUgAAAFoAAAAnCAYAAACPFF8dAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsIAAA7CARUoSoAAAAVLSURBVGhD7Zu7TytHFMbHyxvsJeb9kjCiCihRpJAobaqblFGUJmXqKMofkyZpaKhS5HaJIgUpRZQmoqAhSgEUSBZgsAE/uJhX5ht80LmH8e56fdde7PuTPu0yj/XMt7Nnd2aXxPr6uuKMjIx84LruR47jJGtJbeeVplQqOaOjo+8MDAwk7u7uyrWsWIF2FYvFf3Rbt/HnQ+oDj0Ynk8kPl5eXf9Amf6L7pW5vb9X9/b3Jaye5XE719fWpubk51dPTY/bjijba+KbN3t7d3f324uLir1rWg9HpdPrFysrKy0KhMJTNZtX19XUtu/0sLi6qyclJlUqlcLWpRCJRy4knNzc3ShusKpXKq52dnS/z+fyvSE9sbGxMrq2t/Xd8fJw+PDw0hf1oRWdxNY2Pj6tMJqMmJiZUf3//Y3ocrjQJPOG+nJ2dYWSXt7a23tMRYt+Zn5//rlqteppMB5EHi5rZ2VmEtEeTAUzGJRo3yZOv7ydo94j293v8ndjW6JDxvh7RpoBEGtsKo9FofdNTq6urampqSvX29tZynhcIIUdHR//qUb3iDA4OZnDzs0Gm0khulQCMBs/VZIC2Dw8Pv6v71OvoO7lri3nUYb5tlToRp7Z9Deos37ZanYbVaA7vON/qCU1k6kQC94oMhxFk+FuCU9doPnptkPFRqBN5YjTvKO1LE3iZtwSjMwNiDGnYaD6aEa/1czieFdXQ0JB1wQfPw5C8Cii9Wwg9omHw2NiYmSLDaCz4YNoJ8ScHpGNBCGU4SIe6hVBGY+0BBmOiUy6XzQIKpptY9cOohrESjHg+y+u2ON+w0TAXpgGYfHl5aZYGq9WqMRsLLDDbNnXGyelWQsVoisUwl4OTQGvZPF5TOsxHyOlGQsdogNEroTQZGkqlktkiLnfq7M+LpnpsM4zS5EIVXvFUKhVzAmC2zH+OoA/1JGnYaByEwoN8PONhBXFbgngOw1GvnaNamhJWjdBwb2EmDAP0/EwvTV3XNQbiRNDJ4KBxuIGGQXayGXlhKx9WnFDDCjdBGEZhIJ1Om+dnmI2RXCwWayWfgrpXV1e1v4IhG10P2dEwCoKtnpQkVOgAGNX5fN7c5LCP+IvHOzxT85sk0uUoxt+oh7ygyI7Y5IetTlSSNBUoYSheg8E4mCYf9wDy5asyqlfvFZrE1pFGhd+0pYdRPbzKPTGaF6B9WVEeJGro95uRH7Y6jcqLuiOaKvIDyP2oFBRb3bDywlbeT5LAocPvQFEif5sUBFu9RuVHkDq+RvOK/ECIeW8y7nHZsJULIj9sdRpVEKxGU2W+lftRywtb+bDywlY+qCTGaLkuAagw39pGcBSjWoJJkFe+hJdtRn7Y6kBAznwdZPCVNg5V4gegfS4KI29KgB4VMWVHo7nZtjpcvG1hZTuulK0eID/RdpQDjn7+PcfMrh5UGciDRiVA69w03UfjMdVHw9EB5EUp/IaXbHXQdrwUQTsB2q5nwZc6/T6xubn5WyaT+Wxvb08VCgVTwAtbmIkCNHpmZkYtLCyY76P5iwQ6GXGE/MHMFzPlg4ODP/f39z91Tk9Pfzw/P1dLS0tqenra10h0shUC+JQYbTs5OXltfQRtjKvQdhhMyuVyP5k244t/PXJ+0aPmCywM4dLEohAuD1S0QUa0ApiMD9LxMTrCB1SvXe0GnuHegi1M1m3/I5vNvtBZd8Zo3fCkNvvnZDL5OV41Ic7EqTM48RjReOdo+3QhLmAAwmis4ejQ8bu+Ir/SaWYpk/9XViKVSn3tuu43ujMf67t8975JDYk29UrfAP/WA2NdawNJDzlK/Q9RjPZ1HEiBtwAAAABJRU5ErkJggg=='

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'

color_map = {
    'alice blue': '#F0F8FF',
    'AliceBlue': '#F0F8FF',
    'antique white': '#FAEBD7',
    'AntiqueWhite': '#FAEBD7',
    'AntiqueWhite1': '#FFEFDB',
    'AntiqueWhite2': '#EEDFCC',
    'AntiqueWhite3': '#CDC0B0',
    'AntiqueWhite4': '#8B8378',
    'aquamarine': '#7FFFD4',
    'aquamarine1': '#7FFFD4',
    'aquamarine2': '#76EEC6',
    'aquamarine3': '#66CDAA',
    'aquamarine4': '#458B74',
    'azure': '#F0FFFF',
    'azure1': '#F0FFFF',
    'azure2': '#E0EEEE',
    'azure3': '#C1CDCD',
    'azure4': '#838B8B',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    'bisque1': '#FFE4C4',
    'bisque2': '#EED5B7',
    'bisque3': '#CDB79E',
    'bisque4': '#8B7D6B',
    'black': '#000000',
    'blanched almond': '#FFEBCD',
    'BlanchedAlmond': '#FFEBCD',
    'blue': '#0000FF',
    'blue violet': '#8A2BE2',
    'blue1': '#0000FF',
    'blue2': '#0000EE',
    'blue3': '#0000CD',
    'blue4': '#00008B',
    'BlueViolet': '#8A2BE2',
    'brown': '#A52A2A',
    'brown1': '#FF4040',
    'brown2': '#EE3B3B',
    'brown3': '#CD3333',
    'brown4': '#8B2323',
    'burlywood': '#DEB887',
    'burlywood1': '#FFD39B',
    'burlywood2': '#EEC591',
    'burlywood3': '#CDAA7D',
    'burlywood4': '#8B7355',
    'cadet blue': '#5F9EA0',
    'CadetBlue': '#5F9EA0',
    'CadetBlue1': '#98F5FF',
    'CadetBlue2': '#8EE5EE',
    'CadetBlue3': '#7AC5CD',
    'CadetBlue4': '#53868B',
    'chartreuse': '#7FFF00',
    'chartreuse1': '#7FFF00',
    'chartreuse2': '#76EE00',
    'chartreuse3': '#66CD00',
    'chartreuse4': '#458B00',
    'chocolate': '#D2691E',
    'chocolate1': '#FF7F24',
    'chocolate2': '#EE7621',
    'chocolate3': '#CD661D',
    'chocolate4': '#8B4513',
    'coral': '#FF7F50',
    'coral1': '#FF7256',
    'coral2': '#EE6A50',
    'coral3': '#CD5B45',
    'coral4': '#8B3E2F',
    'cornflower blue': '#6495ED',
    'CornflowerBlue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'cornsilk1': '#FFF8DC',
    'cornsilk2': '#EEE8CD',
    'cornsilk3': '#CDC8B1',
    'cornsilk4': '#8B8878',
    'cyan': '#00FFFF',
    'cyan1': '#00FFFF',
    'cyan2': '#00EEEE',
    'cyan3': '#00CDCD',
    'cyan4': '#008B8B',
    'dark blue': '#00008B',
    'dark cyan': '#008B8B',
    'dark goldenrod': '#B8860B',
    'dark gray': '#A9A9A9',
    'dark green': '#006400',
    'dark grey': '#A9A9A9',
    'dark khaki': '#BDB76B',
    'dark magenta': '#8B008B',
    'dark olive green': '#556B2F',
    'dark orange': '#FF8C00',
    'dark orchid': '#9932CC',
    'dark red': '#8B0000',
    'dark salmon': '#E9967A',
    'dark sea green': '#8FBC8F',
    'dark slate blue': '#483D8B',
    'dark slate gray': '#2F4F4F',
    'dark slate grey': '#2F4F4F',
    'dark turquoise': '#00CED1',
    'dark violet': '#9400D3',
    'DarkBlue': '#00008B',
    'DarkCyan': '#008B8B',
    'DarkGoldenrod': '#B8860B',
    'DarkGoldenrod1': '#FFB90F',
    'DarkGoldenrod2': '#EEAD0E',
    'DarkGoldenrod3': '#CD950C',
    'DarkGoldenrod4': '#8B6508',
    'DarkGray': '#A9A9A9',
    'DarkGreen': '#006400',
    'DarkGrey': '#A9A9A9',
    'DarkKhaki': '#BDB76B',
    'DarkMagenta': '#8B008B',
    'DarkOliveGreen': '#556B2F',
    'DarkOliveGreen1': '#CAFF70',
    'DarkOliveGreen2': '#BCEE68',
    'DarkOliveGreen3': '#A2CD5A',
    'DarkOliveGreen4': '#6E8B3D',
    'DarkOrange': '#FF8C00',
    'DarkOrange1': '#FF7F00',
    'DarkOrange2': '#EE7600',
    'DarkOrange3': '#CD6600',
    'DarkOrange4': '#8B4500',
    'DarkOrchid': '#9932CC',
    'DarkOrchid1': '#BF3EFF',
    'DarkOrchid2': '#B23AEE',
    'DarkOrchid3': '#9A32CD',
    'DarkOrchid4': '#68228B',
    'DarkRed': '#8B0000',
    'DarkSalmon': '#E9967A',
    'DarkSeaGreen': '#8FBC8F',
    'DarkSeaGreen1': '#C1FFC1',
    'DarkSeaGreen2': '#B4EEB4',
    'DarkSeaGreen3': '#9BCD9B',
    'DarkSeaGreen4': '#698B69',
    'DarkSlateBlue': '#483D8B',
    'DarkSlateGray': '#2F4F4F',
    'DarkSlateGray1': '#97FFFF',
    'DarkSlateGray2': '#8DEEEE',
    'DarkSlateGray3': '#79CDCD',
    'DarkSlateGray4': '#528B8B',
    'DarkSlateGrey': '#2F4F4F',
    'DarkTurquoise': '#00CED1',
    'DarkViolet': '#9400D3',
    'deep pink': '#FF1493',
    'deep sky blue': '#00BFFF',
    'DeepPink': '#FF1493',
    'DeepPink1': '#FF1493',
    'DeepPink2': '#EE1289',
    'DeepPink3': '#CD1076',
    'DeepPink4': '#8B0A50',
    'DeepSkyBlue': '#00BFFF',
    'DeepSkyBlue1': '#00BFFF',
    'DeepSkyBlue2': '#00B2EE',
    'DeepSkyBlue3': '#009ACD',
    'DeepSkyBlue4': '#00688B',
    'dim gray': '#696969',
    'dim grey': '#696969',
    'DimGray': '#696969',
    'DimGrey': '#696969',
    'dodger blue': '#1E90FF',
    'DodgerBlue': '#1E90FF',
    'DodgerBlue1': '#1E90FF',
    'DodgerBlue2': '#1C86EE',
    'DodgerBlue3': '#1874CD',
    'DodgerBlue4': '#104E8B',
    'firebrick': '#B22222',
    'firebrick1': '#FF3030',
    'firebrick2': '#EE2C2C',
    'firebrick3': '#CD2626',
    'firebrick4': '#8B1A1A',
    'floral white': '#FFFAF0',
    'FloralWhite': '#FFFAF0',
    'forest green': '#228B22',
    'ForestGreen': '#228B22',
    'gainsboro': '#DCDCDC',
    'ghost white': '#F8F8FF',
    'GhostWhite': '#F8F8FF',
    'gold': '#FFD700',
    'gold1': '#FFD700',
    'gold2': '#EEC900',
    'gold3': '#CDAD00',
    'gold4': '#8B7500',
    'goldenrod': '#DAA520',
    'goldenrod1': '#FFC125',
    'goldenrod2': '#EEB422',
    'goldenrod3': '#CD9B1D',
    'goldenrod4': '#8B6914',
    'green': '#00FF00',
    'green yellow': '#ADFF2F',
    'green1': '#00FF00',
    'green2': '#00EE00',
    'green3': '#00CD00',
    'green4': '#008B00',
    'GreenYellow': '#ADFF2F',
    'grey': '#BEBEBE',
    'grey0': '#000000',
    'grey1': '#030303',
    'grey2': '#050505',
    'grey3': '#080808',
    'grey4': '#0A0A0A',
    'grey5': '#0D0D0D',
    'grey6': '#0F0F0F',
    'grey7': '#121212',
    'grey8': '#141414',
    'grey9': '#171717',
    'grey10': '#1A1A1A',
    'grey11': '#1C1C1C',
    'grey12': '#1F1F1F',
    'grey13': '#212121',
    'grey14': '#242424',
    'grey15': '#262626',
    'grey16': '#292929',
    'grey17': '#2B2B2B',
    'grey18': '#2E2E2E',
    'grey19': '#303030',
    'grey20': '#333333',
    'grey21': '#363636',
    'grey22': '#383838',
    'grey23': '#3B3B3B',
    'grey24': '#3D3D3D',
    'grey25': '#404040',
    'grey26': '#424242',
    'grey27': '#454545',
    'grey28': '#474747',
    'grey29': '#4A4A4A',
    'grey30': '#4D4D4D',
    'grey31': '#4F4F4F',
    'grey32': '#525252',
    'grey33': '#545454',
    'grey34': '#575757',
    'grey35': '#595959',
    'grey36': '#5C5C5C',
    'grey37': '#5E5E5E',
    'grey38': '#616161',
    'grey39': '#636363',
    'grey40': '#666666',
    'grey41': '#696969',
    'grey42': '#6B6B6B',
    'grey43': '#6E6E6E',
    'grey44': '#707070',
    'grey45': '#737373',
    'grey46': '#757575',
    'grey47': '#787878',
    'grey48': '#7A7A7A',
    'grey49': '#7D7D7D',
    'grey50': '#7F7F7F',
    'grey51': '#828282',
    'grey52': '#858585',
    'grey53': '#878787',
    'grey54': '#8A8A8A',
    'grey55': '#8C8C8C',
    'grey56': '#8F8F8F',
    'grey57': '#919191',
    'grey58': '#949494',
    'grey59': '#969696',
    'grey60': '#999999',
    'grey61': '#9C9C9C',
    'grey62': '#9E9E9E',
    'grey63': '#A1A1A1',
    'grey64': '#A3A3A3',
    'grey65': '#A6A6A6',
    'grey66': '#A8A8A8',
    'grey67': '#ABABAB',
    'grey68': '#ADADAD',
    'grey69': '#B0B0B0',
    'grey70': '#B3B3B3',
    'grey71': '#B5B5B5',
    'grey72': '#B8B8B8',
    'grey73': '#BABABA',
    'grey74': '#BDBDBD',
    'grey75': '#BFBFBF',
    'grey76': '#C2C2C2',
    'grey77': '#C4C4C4',
    'grey78': '#C7C7C7',
    'grey79': '#C9C9C9',
    'grey80': '#CCCCCC',
    'grey81': '#CFCFCF',
    'grey82': '#D1D1D1',
    'grey83': '#D4D4D4',
    'grey84': '#D6D6D6',
    'grey85': '#D9D9D9',
    'grey86': '#DBDBDB',
    'grey87': '#DEDEDE',
    'grey88': '#E0E0E0',
    'grey89': '#E3E3E3',
    'grey90': '#E5E5E5',
    'grey91': '#E8E8E8',
    'grey92': '#EBEBEB',
    'grey93': '#EDEDED',
    'grey94': '#F0F0F0',
    'grey95': '#F2F2F2',
    'grey96': '#F5F5F5',
    'grey97': '#F7F7F7',
    'grey98': '#FAFAFA',
    'grey99': '#FCFCFC',
    'grey100': '#FFFFFF',
    'honeydew': '#F0FFF0',
    'honeydew1': '#F0FFF0',
    'honeydew2': '#E0EEE0',
    'honeydew3': '#C1CDC1',
    'honeydew4': '#838B83',
    'hot pink': '#FF69B4',
    'HotPink': '#FF69B4',
    'HotPink1': '#FF6EB4',
    'HotPink2': '#EE6AA7',
    'HotPink3': '#CD6090',
    'HotPink4': '#8B3A62',
    'indian red': '#CD5C5C',
    'IndianRed': '#CD5C5C',
    'IndianRed1': '#FF6A6A',
    'IndianRed2': '#EE6363',
    'IndianRed3': '#CD5555',
    'IndianRed4': '#8B3A3A',
    'ivory': '#FFFFF0',
    'ivory1': '#FFFFF0',
    'ivory2': '#EEEEE0',
    'ivory3': '#CDCDC1',
    'ivory4': '#8B8B83',
    'khaki': '#F0E68C',
    'khaki1': '#FFF68F',
    'khaki2': '#EEE685',
    'khaki3': '#CDC673',
    'khaki4': '#8B864E',
    'lavender': '#E6E6FA',
    'lavender blush': '#FFF0F5',
    'LavenderBlush': '#FFF0F5',
    'LavenderBlush1': '#FFF0F5',
    'LavenderBlush2': '#EEE0E5',
    'LavenderBlush3': '#CDC1C5',
    'LavenderBlush4': '#8B8386',
    'lawn green': '#7CFC00',
    'LawnGreen': '#7CFC00',
    'lemon chiffon': '#FFFACD',
    'LemonChiffon': '#FFFACD',
    'LemonChiffon1': '#FFFACD',
    'LemonChiffon2': '#EEE9BF',
    'LemonChiffon3': '#CDC9A5',
    'LemonChiffon4': '#8B8970',
    'light blue': '#ADD8E6',
    'light coral': '#F08080',
    'light cyan': '#E0FFFF',
    'light goldenrod': '#EEDD82',
    'light goldenrod yellow': '#FAFAD2',
    'light gray': '#D3D3D3',
    'light green': '#90EE90',
    'light grey': '#D3D3D3',
    'light pink': '#FFB6C1',
    'light salmon': '#FFA07A',
    'light sea green': '#20B2AA',
    'light sky blue': '#87CEFA',
    'light slate blue': '#8470FF',
    'light slate gray': '#778899',
    'light slate grey': '#778899',
    'light steel blue': '#B0C4DE',
    'light yellow': '#FFFFE0',
    'LightBlue': '#ADD8E6',
    'LightBlue1': '#BFEFFF',
    'LightBlue2': '#B2DFEE',
    'LightBlue3': '#9AC0CD',
    'LightBlue4': '#68838B',
    'LightCoral': '#F08080',
    'LightCyan': '#E0FFFF',
    'LightCyan1': '#E0FFFF',
    'LightCyan2': '#D1EEEE',
    'LightCyan3': '#B4CDCD',
    'LightCyan4': '#7A8B8B',
    'LightGoldenrod': '#EEDD82',
    'LightGoldenrod1': '#FFEC8B',
    'LightGoldenrod2': '#EEDC82',
    'LightGoldenrod3': '#CDBE70',
    'LightGoldenrod4': '#8B814C',
    'LightGoldenrodYellow': '#FAFAD2',
    'LightGray': '#D3D3D3',
    'LightGreen': '#90EE90',
    'LightGrey': '#D3D3D3',
    'LightPink': '#FFB6C1',
    'LightPink1': '#FFAEB9',
    'LightPink2': '#EEA2AD',
    'LightPink3': '#CD8C95',
    'LightPink4': '#8B5F65',
    'LightSalmon': '#FFA07A',
    'LightSalmon1': '#FFA07A',
    'LightSalmon2': '#EE9572',
    'LightSalmon3': '#CD8162',
    'LightSalmon4': '#8B5742',
    'LightSeaGreen': '#20B2AA',
    'LightSkyBlue': '#87CEFA',
    'LightSkyBlue1': '#B0E2FF',
    'LightSkyBlue2': '#A4D3EE',
    'LightSkyBlue3': '#8DB6CD',
    'LightSkyBlue4': '#607B8B',
    'LightSlateBlue': '#8470FF',
    'LightSlateGray': '#778899',
    'LightSlateGrey': '#778899',
    'LightSteelBlue': '#B0C4DE',
    'LightSteelBlue1': '#CAE1FF',
    'LightSteelBlue2': '#BCD2EE',
    'LightSteelBlue3': '#A2B5CD',
    'LightSteelBlue4': '#6E7B8B',
    'LightYellow': '#FFFFE0',
    'LightYellow1': '#FFFFE0',
    'LightYellow2': '#EEEED1',
    'LightYellow3': '#CDCDB4',
    'LightYellow4': '#8B8B7A',
    'lime green': '#32CD32',
    'LimeGreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'magenta1': '#FF00FF',
    'magenta2': '#EE00EE',
    'magenta3': '#CD00CD',
    'magenta4': '#8B008B',
    'maroon': '#B03060',
    'maroon1': '#FF34B3',
    'maroon2': '#EE30A7',
    'maroon3': '#CD2990',
    'maroon4': '#8B1C62',
    'medium aquamarine': '#66CDAA',
    'medium blue': '#0000CD',
    'medium orchid': '#BA55D3',
    'medium purple': '#9370DB',
    'medium sea green': '#3CB371',
    'medium slate blue': '#7B68EE',
    'medium spring green': '#00FA9A',
    'medium turquoise': '#48D1CC',
    'medium violet red': '#C71585',
    'MediumAquamarine': '#66CDAA',
    'MediumBlue': '#0000CD',
    'MediumOrchid': '#BA55D3',
    'MediumOrchid1': '#E066FF',
    'MediumOrchid2': '#D15FEE',
    'MediumOrchid3': '#B452CD',
    'MediumOrchid4': '#7A378B',
    'MediumPurple': '#9370DB',
    'MediumPurple1': '#AB82FF',
    'MediumPurple2': '#9F79EE',
    'MediumPurple3': '#8968CD',
    'MediumPurple4': '#5D478B',
    'MediumSeaGreen': '#3CB371',
    'MediumSlateBlue': '#7B68EE',
    'MediumSpringGreen': '#00FA9A',
    'MediumTurquoise': '#48D1CC',
    'MediumVioletRed': '#C71585',
    'midnight blue': '#191970',
    'MidnightBlue': '#191970',
    'mint cream': '#F5FFFA',
    'MintCream': '#F5FFFA',
    'misty rose': '#FFE4E1',
    'MistyRose': '#FFE4E1',
    'MistyRose1': '#FFE4E1',
    'MistyRose2': '#EED5D2',
    'MistyRose3': '#CDB7B5',
    'MistyRose4': '#8B7D7B',
    'moccasin': '#FFE4B5',
    'navajo white': '#FFDEAD',
    'NavajoWhite': '#FFDEAD',
    'NavajoWhite1': '#FFDEAD',
    'NavajoWhite2': '#EECFA1',
    'NavajoWhite3': '#CDB38B',
    'NavajoWhite4': '#8B795E',
    'navy': '#000080',
    'navy blue': '#000080',
    'NavyBlue': '#000080',
    'old lace': '#FDF5E6',
    'OldLace': '#FDF5E6',
    'olive drab': '#6B8E23',
    'OliveDrab': '#6B8E23',
    'OliveDrab1': '#C0FF3E',
    'OliveDrab2': '#B3EE3A',
    'OliveDrab3': '#9ACD32',
    'OliveDrab4': '#698B22',
    'orange': '#FFA500',
    'orange red': '#FF4500',
    'orange1': '#FFA500',
    'orange2': '#EE9A00',
    'orange3': '#CD8500',
    'orange4': '#8B5A00',
    'OrangeRed': '#FF4500',
    'OrangeRed1': '#FF4500',
    'OrangeRed2': '#EE4000',
    'OrangeRed3': '#CD3700',
    'OrangeRed4': '#8B2500',
    'orchid': '#DA70D6',
    'orchid1': '#FF83FA',
    'orchid2': '#EE7AE9',
    'orchid3': '#CD69C9',
    'orchid4': '#8B4789',
    'pale goldenrod': '#EEE8AA',
    'pale green': '#98FB98',
    'pale turquoise': '#AFEEEE',
    'pale violet red': '#DB7093',
    'PaleGoldenrod': '#EEE8AA',
    'PaleGreen': '#98FB98',
    'PaleGreen1': '#9AFF9A',
    'PaleGreen2': '#90EE90',
    'PaleGreen3': '#7CCD7C',
    'PaleGreen4': '#548B54',
    'PaleTurquoise': '#AFEEEE',
    'PaleTurquoise1': '#BBFFFF',
    'PaleTurquoise2': '#AEEEEE',
    'PaleTurquoise3': '#96CDCD',
    'PaleTurquoise4': '#668B8B',
    'PaleVioletRed': '#DB7093',
    'PaleVioletRed1': '#FF82AB',
    'PaleVioletRed2': '#EE799F',
    'PaleVioletRed3': '#CD687F',
    'PaleVioletRed4': '#8B475D',
    'papaya whip': '#FFEFD5',
    'PapayaWhip': '#FFEFD5',
    'peach puff': '#FFDAB9',
    'PeachPuff': '#FFDAB9',
    'PeachPuff1': '#FFDAB9',
    'PeachPuff2': '#EECBAD',
    'PeachPuff3': '#CDAF95',
    'PeachPuff4': '#8B7765',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'pink1': '#FFB5C5',
    'pink2': '#EEA9B8',
    'pink3': '#CD919E',
    'pink4': '#8B636C',
    'plum': '#DDA0DD',
    'plum1': '#FFBBFF',
    'plum2': '#EEAEEE',
    'plum3': '#CD96CD',
    'plum4': '#8B668B',
    'powder blue': '#B0E0E6',
    'PowderBlue': '#B0E0E6',
    'purple': '#A020F0',
    'purple1': '#9B30FF',
    'purple2': '#912CEE',
    'purple3': '#7D26CD',
    'purple4': '#551A8B',
    'red': '#FF0000',
    'red1': '#FF0000',
    'red2': '#EE0000',
    'red3': '#CD0000',
    'red4': '#8B0000',
    'rosy brown': '#BC8F8F',
    'RosyBrown': '#BC8F8F',
    'RosyBrown1': '#FFC1C1',
    'RosyBrown2': '#EEB4B4',
    'RosyBrown3': '#CD9B9B',
    'RosyBrown4': '#8B6969',
    'royal blue': '#4169E1',
    'RoyalBlue': '#4169E1',
    'RoyalBlue1': '#4876FF',
    'RoyalBlue2': '#436EEE',
    'RoyalBlue3': '#3A5FCD',
    'RoyalBlue4': '#27408B',
    'saddle brown': '#8B4513',
    'SaddleBrown': '#8B4513',
    'salmon': '#FA8072',
    'salmon1': '#FF8C69',
    'salmon2': '#EE8262',
    'salmon3': '#CD7054',
    'salmon4': '#8B4C39',
    'sandy brown': '#F4A460',
    'SandyBrown': '#F4A460',
    'sea green': '#2E8B57',
    'SeaGreen': '#2E8B57',
    'SeaGreen1': '#54FF9F',
    'SeaGreen2': '#4EEE94',
    'SeaGreen3': '#43CD80',
    'SeaGreen4': '#2E8B57',
    'seashell': '#FFF5EE',
    'seashell1': '#FFF5EE',
    'seashell2': '#EEE5DE',
    'seashell3': '#CDC5BF',
    'seashell4': '#8B8682',
    'sienna': '#A0522D',
    'sienna1': '#FF8247',
    'sienna2': '#EE7942',
    'sienna3': '#CD6839',
    'sienna4': '#8B4726',
    'sky blue': '#87CEEB',
    'SkyBlue': '#87CEEB',
    'SkyBlue1': '#87CEFF',
    'SkyBlue2': '#7EC0EE',
    'SkyBlue3': '#6CA6CD',
    'SkyBlue4': '#4A708B',
    'slate blue': '#6A5ACD',
    'slate gray': '#708090',
    'slate grey': '#708090',
    'SlateBlue': '#6A5ACD',
    'SlateBlue1': '#836FFF',
    'SlateBlue2': '#7A67EE',
    'SlateBlue3': '#6959CD',
    'SlateBlue4': '#473C8B',
    'SlateGray': '#708090',
    'SlateGray1': '#C6E2FF',
    'SlateGray2': '#B9D3EE',
    'SlateGray3': '#9FB6CD',
    'SlateGray4': '#6C7B8B',
    'SlateGrey': '#708090',
    'snow': '#FFFAFA',
    'snow1': '#FFFAFA',
    'snow2': '#EEE9E9',
    'snow3': '#CDC9C9',
    'snow4': '#8B8989',
    'spring green': '#00FF7F',
    'SpringGreen': '#00FF7F',
    'SpringGreen1': '#00FF7F',
    'SpringGreen2': '#00EE76',
    'SpringGreen3': '#00CD66',
    'SpringGreen4': '#008B45',
    'steel blue': '#4682B4',
    'SteelBlue': '#4682B4',
    'SteelBlue1': '#63B8FF',
    'SteelBlue2': '#5CACEE',
    'SteelBlue3': '#4F94CD',
    'SteelBlue4': '#36648B',
    'tan': '#D2B48C',
    'tan1': '#FFA54F',
    'tan2': '#EE9A49',
    'tan3': '#CD853F',
    'tan4': '#8B5A2B',
    'thistle': '#D8BFD8',
    'thistle1': '#FFE1FF',
    'thistle2': '#EED2EE',
    'thistle3': '#CDB5CD',
    'thistle4': '#8B7B8B',
    'tomato': '#FF6347',
    'tomato1': '#FF6347',
    'tomato2': '#EE5C42',
    'tomato3': '#CD4F39',
    'tomato4': '#8B3626',
    'turquoise': '#40E0D0',
    'turquoise1': '#00F5FF',
    'turquoise2': '#00E5EE',
    'turquoise3': '#00C5CD',
    'turquoise4': '#00868B',
    'violet': '#EE82EE',
    'violet red': '#D02090',
    'VioletRed': '#D02090',
    'VioletRed1': '#FF3E96',
    'VioletRed2': '#EE3A8C',
    'VioletRed3': '#CD3278',
    'VioletRed4': '#8B2252',
    'wheat': '#F5DEB3',
    'wheat1': '#FFE7BA',
    'wheat2': '#EED8AE',
    'wheat3': '#CDBA96',
    'wheat4': '#8B7E66',
    'white': '#FFFFFF',
    'white smoke': '#F5F5F5',
    'WhiteSmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellow green': '#9ACD32',
    'yellow1': '#FFFF00',
    'yellow2': '#EEEE00',
    'yellow3': '#CDCD00',
    'yellow4': '#8B8B00',
    'YellowGreen': '#9ACD32',
}

w_kws = {
    'finalize': True,
    'resizable': True,
    'default_button_element_size': (6, 1),
    'default_element_size': (14, 1),
    'font': ('size', 8),
    # 'auto_size_text' : True,
    # 'auto_size_buttons' : True,
}

b_kws = {'font': ('size', 6)}

b3_kws = {'font': ('size', 6),
          'size': (3, 1)}

b6_kws = {'font': ('size', 6),
          'size': (6, 1)}

b12_kws = {'font': ('size', 6),
           'size': (12, 1)}

t8_kws = {'size': (8, 1)
          }

t10_kws = {'size': (10, 1)
           }
t12_kws = {'size': (12, 1)
           }

t14_kws = {'size': (14, 1)}

t40_kws = {'size': (40, 1)}

t5_kws = {'size': (5, 1)}
t2_kws = {'size': (2, 1)}
t24_kws = {'size': (24, 1)}


# sg.theme('LightGreen')
def popup_color_chooser(look_and_feel=None):
    """

    :return: Any(str, None) Returns hex string of color chosen or None if nothing was chosen
    """

    old_look_and_feel = None
    if look_and_feel is not None:
        old_look_and_feel = sg.CURRENT_LOOK_AND_FEEL
        sg.change_look_and_feel(look_and_feel)

    button_size = (1, 1)

    def ColorButton(color):
        """
        A User Defined Element - returns a Button that configured in a certain way.
        :param color: Tuple[str, str] ( color name, hex string)
        :return: sg.Button object
        """
        return sg.B(button_color=('white', color[1]), pad=(0, 0), size=button_size, key=color,
                    tooltip=f'{color[0]}:{color[1]}', border_width=0)

    N = len(list(color_map.keys()))
    row_len = 40

    grid = [[ColorButton(list(color_map.items())[c + j * row_len]) for c in range(0, row_len)] for j in
            range(0, N // row_len)]
    grid += [[ColorButton(list(color_map.items())[c + N - N % row_len]) for c in range(0, N % row_len)]]

    layout = [[sg.Text('Pick a color', font='Def 18')]] + grid + \
             [[sg.Button('OK'), sg.T(size=(30, 1), key='-OUT-'), sg.Button('Cancel'), sg.T(size=(30, 1))]]

    window = sg.Window('Window Title', layout, no_titlebar=True, grab_anywhere=True, keep_on_top=True,
                       use_ttk_buttons=True)
    color_chosen = None
    while True:  # Event Loop
        event, values = window.read()
        if event in (None, 'Cancel', 'OK'):
            if event in (None, 'Cancel'):
                color_chosen = None
            break
        window['-OUT-'](f'You chose {event[0]} : {event[1]}')
        color_chosen = event[0]
        # color_chosen = event[1]
    window.close()
    if old_look_and_feel is not None:
        sg.change_look_and_feel(old_look_and_feel)
    return color_chosen


def color_pick_layout(name, color=None):
    return [sg.T('', **t5_kws), sg.T('color', **t5_kws),
            sg.Combo(list(color_map.keys()), default_value=color, k=f'{name}_color', enable_events=True, readonly=True,
                     **t8_kws),
            sg.B('Pick', k=f'PICK {name}_color', **b3_kws)]


def retrieve_value(v, t):
    if v in ['', 'None', None]:
        vv = None
    elif v in ['sample', 'fit']:
        vv = v
    elif t in ['bool', bool]:
        if v in ['False', False, 0, '0']:
            vv = False
        elif v in ['True', True, 1, '1']:
            vv = True
    elif t in ['float', float]:
        vv = float(v)
    elif t in ['str', str]:
        vv = str(v)
    elif t in ['int', int]:
        vv = int(v)
    elif type(v) == t:
        vv = v
    elif t == List[Tuple[float, float]]:
        if type(v) == str:
            v = v.replace('{', ' ')
            v = v.replace('}', ' ')
            v = v.replace('[', ' ')
            v = v.replace(']', ' ')
            v = v.replace('(', ' ')
            v = v.replace(')', ' ')
            v = v.replace(',', ' ')
            vv = [tuple([float(x) for x in t.split()]) for t in v.split('   ')]
        elif type(v) == list:
            vv = v
    elif t == Tuple[float, float] and type(v) == str:
        v = v.replace('{', '')
        v = v.replace('}', '')
        v = v.replace('[', '')
        v = v.replace(']', '')
        v = v.replace('(', '')
        v = v.replace(')', '')
        v = v.replace("'", '')
        v = v.replace(",", ' ')
        vv = tuple([float(x) for x in v.split()])
    elif t == Type and type(v) == str:
        if 'str' in v:
            vv = str
        elif 'float' in v:
            vv = float
        elif 'bool' in v:
            vv = bool
        elif 'int' in v:
            vv = int

    elif t == tuple or t == list:
        try:
            vv = literal_eval(v)
        except:
            vv = [float(x) for x in v.split()]
            if t == tuple:
                vv = tuple(vv)

    # elif type(t) == dict:
    #     vv = v
    elif type(t) == list:
        vv = retrieve_value(v, type(t[0]))
        if vv not in t:
            raise ValueError(f'Retrieved value {vv} not in list {t}')
    else:
        vv = v
    return vv


def retrieve_dict(dic, type_dic):
    return {k: retrieve_value(v, type_dic[k]) for k, v in dic.items()}


def get_table_data(values, pars_dict, Nagents):
    data = []
    for i in range(Nagents):
        dic = {}
        for j, (p, t) in enumerate(pars_dict.items()):
            v = values[(i, p)]
            dic[p] = retrieve_value(v, t)
        data.append(dic)
    return data


def set_agent_dict(dic, type_dic, header='unique_id', title='Agent list'):
    t0 = fun.agent_dict2list(dic, header=header)
    t1 = gui_table(t0, type_dic, title=title)
    dic = fun.agent_list2dict(t1, header=header)
    return dic


def build_table_window(data, pars_dict, title, return_layout=False):
    t12_kws_c = {**t12_kws,
                 'justification': 'center'}

    pars = list(pars_dict.keys())
    par_types = list(pars_dict.values())
    Nagents, Npars = len(data), len(pars)
    # A HIGHLY unusual layout definition
    # Normally a layout is specified 1 ROW at a time. Here multiple rows are being contatenated together to produce the layout
    # Note the " + \ " at the ends of the lines rather than the usual " , "
    # This is done because each line is a list of lists
    layout = [[sg.Text(' ', **t2_kws)] + [sg.Text(p, key=p, enable_events=True, **t12_kws_c) for p in pars]] + \
             [
                 [sg.T(i + 1, **t2_kws)] +
                 [sg.Input(data[i][p], key=(i, p), **t12_kws_c) if type(pars_dict[p]) != list else sg.Combo(
                     pars_dict[p], default_value=data[i][p], key=(i, p), enable_events=True, readonly=True,
                     **t12_kws) for p in pars] for i in range(Nagents)] + \
             [[sg.Button('Add', **b6_kws), sg.Button('Remove', **b6_kws),
               sg.Button('Ok', **b6_kws), sg.Button('Cancel', **b6_kws)]]

    if return_layout:
        return layout

    # Create the window
    table_window = sg.Window(title, layout, default_element_size=(20, 1), element_padding=(1, 1),
                             return_keyboard_events=True, finalize=True)
    table_window.close_destroys_window = True
    return Nagents, Npars, pars, table_window


# def immutable_table(name, dic):
#     if name == 'Source groups':
#         print(dic)
#         headings = ['group', 'N', 'default_color']
#         data=np.ones([len(dic), len(headings)])*np.nan
#         for i, id, pars in enumerate(dic.items()):
#             data[i][0] = id
#             for j, p in enumerate(headings[1:]) :
#                 for k,v in pars.items() :
#                     if k==p :
#                         data[i][j]=v
#     layout = [[sg.Table(values=data, headings=headings, max_col_width=25, background_color='lightblue',
#                         auto_size_columns=True,
#                         display_row_numbers=True,
#                         justification='right',
#                         font=w_kws['font'],
#                         # num_rows=20,
#                         alternating_row_color='lightyellow',
#                         key=f'TABLE_{name}'
#                         # tooltip='This is a table'
#                )],
#               # [sg.Button('Read'), sg.Button('Double'), sg.Button('Change Colors')],
#               # [sg.Text('Read = read which rows are selected')],
#               # [sg.Text('Double = double the amount of data in the table')],
#               # [sg.Text('Change Colors = Changes the colors of rows 8 and 9')]
#               ]
#     return layout


def gui_table(data, pars_dict, title='Agent list'):
    """
        Another simple table created from Input Text Elements.  This demo adds the ability to "navigate" around the drawing using
        the arrow keys. The tab key works automatically, but the arrow keys are done in the code below.
    """

    sg.change_look_and_feel('Dark Brown 2')  # No excuse for gray windows
    # Show a "splash" type message so the user doesn't give up waiting
    sg.popup_quick_message('Hang on for a moment, this will take a bit to create....', auto_close=True,
                           non_blocking=True)

    Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)
    print(Nagents, Npars, pars)
    current_cell = (0, 0)
    while True:  # Event Loop
        event, values = table_window.read()
        if event in (None, 'Cancel'):
            table_window.close()
            return data
            # break
        if event == 'Ok':
            data = get_table_data(values, pars_dict, Nagents)
            table_window.close()
            return data
        elem = table_window.find_element_with_focus()
        current_cell = elem.Key if elem and type(elem.Key) is tuple else (0, 0)
        r, c = current_cell

        if event.startswith('Down'):
            r = r + 1 * (r < Nagents - 1)
        elif event.startswith('Left'):
            c = c - 1 * (c > 0)
        elif event.startswith('Right'):
            c = c + 1 * (c < Npars - 1)
        elif event.startswith('Up'):
            r = r - 1 * (r > 0)
        elif event in pars:  # Perform a sort if a column heading was clicked
            col_clicked = pars.index(event)
            try:
                table = [[int(values[(row, col)]) for col in range(Npars)] for row in range(Nagents)]
                new_table = sorted(table, key=operator.itemgetter(col_clicked))
            except:
                sg.popup_error('Error in table', 'Your table must contain only ints if you wish to sort by column')
            else:
                for i in range(Nagents):
                    for j in range(Npars):
                        table_window[(i, j)].update(new_table[i][j])
                [table_window[c].update(font='Any 14') for c in pars]  # make all column headings be normal fonts
                table_window[event].update(font='Any 14 bold')  # bold the font that was clicked
        # if the current cell changed, set focus on new cell
        if current_cell != (r, c):
            current_cell = r, c
            table_window[current_cell].set_focus()  # set the focus on the element moved to
            table_window[current_cell].update(
                select=True)  # when setting focus, also highlight the data in the element so typing overwrites
        if event == 'Add':
            data = get_table_data(values, pars_dict, Nagents)
            try:
                new_row = data[r]
            except:
                new_row = {k: None for k in pars_dict.keys()}
            data.append(new_row)
            table_window.close()
            Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)
        elif event == 'Remove':
            data = get_table_data(values, pars_dict, Nagents)
            data = [d for i, d in enumerate(data) if i != r]
            table_window.close()
            Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)
            # table_window.close()
            # gui_table(data, pars_dict, title='Agent list')

    # if clicked button to dump the table's values
    # if event.startswith('Show Table'):
    #     table = [[values[(row, col)] for col in range(Npars)] for row in range(Nagents)]
    #     sg.popup_scrolled('your_table = [ ', ',\n'.join([str(table[i]) for i in range(Nagents)]) + '  ]', title='Copy your data from here')


def update_window_from_dict(window, dic, prefix=None):
    if dic is not None:
        for k, v in dic.items():
            if prefix is not None:
                k = f'{prefix}_{k}'
            if type(v) == bool:
                window[f'TOGGLE_{k}'].metadata.state = v
                window[f'TOGGLE_{k}'].update(image_data=on_image if v else off_image)
            elif type(v) == dict:
                if prefix is not None:
                    new_prefix = k
                else:
                    new_prefix = None
                update_window_from_dict(window, v, prefix=new_prefix)
            elif v is None:
                window.Element(k).Update(value='')
            else:
                window.Element(k).Update(value=v)


class SectionDict:
    def __init__(self, name, dict, type_dict=None, toggled_subsections=True):
        self.init_dict = dict
        # self.named_init_dict=self.named_dict(dict)
        self.type_dict = type_dict
        self.toggled_subsections = toggled_subsections
        self.name = name
        self.subdicts = {}

    def init_section(self):
        if self.init_dict is not None:
            dic = self.init_dict
        else:
            return []
        l = []
        for k, v in dic.items():
            k0 = f'{self.name}_{k}'
            if type(v) == bool:
                l.append(named_bool_button(k, v, k0))
            elif type(v) == dict:
                if self.type_dict is not None:
                    type_dict = self.type_dict[k]
                else:
                    type_dict = None
                self.subdicts[k0] = CollapsibleDict(k0, True, disp_name=k, dict=v, type_dict=type_dict,
                                                    toggle=self.toggled_subsections)
                ll = self.subdicts[k0].get_section()
                l.append(ll)
            else:
                temp = sg.In(v, key=k0)
                if self.type_dict is not None:
                    if type(self.type_dict[k]) == list:
                        temp = sg.Combo(self.type_dict[k], default_value=v, key=k0, enable_events=True, readonly=True)
                l.append([sg.Text(f'{k}:'), temp])
        return l

    def get_dict(self, values, window):
        new_dict = copy.deepcopy(self.init_dict)
        if new_dict is None:
            return new_dict
        if self.type_dict is None:
            for i, (k, v) in enumerate(new_dict.items()):
                k0 = f'{self.name}_{k}'
                if type(v) == bool:
                    new_dict[k] = window[f'TOGGLE_{k0}'].metadata.state
                elif type(v) == dict:
                    new_dict[k] = self.subdicts[k0].get_dict(values, window)
                else:
                    vv = values[k0]
                    vv = retrieve_value(vv, type(v))
                    new_dict[k] = vv

        else:
            for i, (k, t) in enumerate(self.type_dict.items()):
                k0 = f'{self.name}_{k}'
                if t == bool:
                    new_dict[k] = window[f'TOGGLE_{k0}'].metadata.state
                elif t == dict or type(t) == dict:
                    new_dict[k] = self.subdicts[k0].get_dict(values, window)
                else:
                    new_dict[k] = retrieve_value(values[k0], t)
        return new_dict

    def get_subdicts(self):
        subdicts = {}
        for s in list(self.subdicts.values()):
            subdicts.update(s.get_subdicts())
        return subdicts


def named_bool_button(name, state, toggle_name=None):
    if toggle_name is None:
        toggle_name = name
    l = [sg.Text(f'{name} :'), bool_button(toggle_name, state)]
    return l


def bool_button(name, state, disabled=False):
    if state:
        if disabled:
            image = on_image_disabled
        else:
            image = on_image
    elif state == False:
        if disabled:
            image = off_image_disabled
        else:
            image = off_image
    elif state is None:
        image = off_image_disabled
    b = sg.Button(image_data=image, k=f'TOGGLE_{name}', border_width=0,
                  button_color=(sg.theme_background_color(), sg.theme_background_color()),
                  disabled_button_color=(sg.theme_background_color(), sg.theme_background_color()),
                  metadata=BtnInfo(state=state), **b6_kws)
    return b


def named_list_layout(text, key, choices, readonly=True, enable_events=True):
    l = [sg.Text(text), sg.Combo(choices, key=key, enable_events=enable_events, readonly=readonly)]
    return l


class Collapsible:
    def __init__(self, name, state, content, disp_name=None, toggle=None, disabled=False, next_to_header=None):
        self.name = name
        if disp_name is None:
            disp_name = name
        self.disp_name = disp_name
        self.state = state
        self.toggle = toggle
        self.symbol = SYMBOL_DOWN if state else SYMBOL_UP
        header = [sg.T(self.symbol, enable_events=True, k=f'OPEN SEC {name}', text_color='black'),
                  sg.T(disp_name, enable_events=True, text_color='black', k=f'SEC {name} TEXT', **t12_kws)]
        if toggle is not None:
            header.append(bool_button(name, toggle, disabled))
        if next_to_header is not None:
            header += next_to_header
        self.section = [header, [collapse(content, f'SEC {name}', visible=state)]]

    def get_section(self, as_col=True):
        if as_col:
            return [sg.Col(self.section)]
        else:
            return self.section

    def set_section(self, section):
        self.section = section

    def update(self, window, dict, use_prefix=True):
        if dict is None:
            self.disable(window)
        else:
            self.enable(window)
            if use_prefix:
                prefix = self.name
            else:
                prefix = None
            update_window_from_dict(window, dict, prefix=prefix)
        return window

    def disable(self, window):
        if self.toggle is not None:
            window[f'TOGGLE_{self.name}'].metadata.state = None
            window[f'TOGGLE_{self.name}'].update(image_data=off_image_disabled)
        self.state = None
        window[f'OPEN SEC {self.name}'].update(SYMBOL_UP)
        window[f'SEC {self.name}'].update(visible=False)

    def enable(self, window):
        if self.toggle is not None:
            window[f'TOGGLE_{self.name}'].update(image_data=on_image_disabled)
        if self.state is None:
            self.state = True
        window[f'OPEN SEC {self.name}'].update(SYMBOL_DOWN)
        window[f'SEC {self.name}'].update(visible=True)


class CollapsibleDict(Collapsible):
    def __init__(self, name, state, dict, dict_name=None, type_dict=None, toggled_subsections=True, **kwargs):
        if dict_name is None:
            dict_name = name
        self.dict_name = dict_name
        self.sectiondict = SectionDict(name=dict_name, dict=dict, type_dict=type_dict,
                                       toggled_subsections=toggled_subsections)
        content = self.sectiondict.init_section()
        super().__init__(name, state, content, **kwargs)

    def get_dict(self, values, window):
        if self.state is None:
            return None
        else:
            return self.sectiondict.get_dict(values, window)

    def get_subdicts(self):
        subdicts = {}
        subdicts[self.name] = self
        all_subdicts = {**subdicts, **self.sectiondict.get_subdicts()}
        return all_subdicts


def collapse(layout, key, visible=True):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key, visible=visible))


def set_kwargs(kwargs, title='Arguments', type_dict=None):
    sec_dict = SectionDict(name=title, dict=kwargs, type_dict=type_dict)
    layout = sec_dict.init_section()
    layout.append([sg.Ok(), sg.Cancel()])
    window = sg.Window(title, layout)
    while True:
        event, values = window.read()
        if event == 'Ok':
            new_kwargs = sec_dict.get_dict(values, window)
            break
        elif event == 'Cancel':
            new_kwargs = kwargs
            break
        elif 'TOGGLE' in event:
            if window[event].metadata.state is not None:
                window[event].metadata.state = not window[event].metadata.state
                window[event].update(image_data=on_image if window[event].metadata.state else off_image)
    window.close()
    del sec_dict
    return new_kwargs

    # if kwargs != {}:
    #     layout = []
    #     for i, (k, v) in enumerate(kwargs.items()):
    #         if not type(v) == dict and not type(v) == np.ndarray:
    #             layout.append([sg.Text(k, size=(20, 1)), sg.Input(default_text=str(v), k=f'KW_{i}', size=(20, 1))])
    #     layout.append([sg.Ok(), sg.Cancel()])
    #     event, values = sg.Window(title, layout).read(close=True)
    #     if event == 'Ok':
    #         for i, (k, v) in enumerate(kwargs.items()):
    #             if type(v) == np.ndarray:
    #                 continue
    #             if not type(v) == dict:
    #                 vv = values[f'KW_{i}']
    #                 if type(v) == bool:
    #                     if vv == 'False':
    #                         vv = False
    #                     elif vv == 'True':
    #                         vv = True
    #                 elif type(v) == list or type(v) == tuple:
    #                     vv = literal_eval(vv)
    #
    #                 elif v is None:
    #                     if vv == 'None':
    #                         vv = None
    #                     else:
    #                         vv = vv
    #                 else:
    #                     vv = type(v)(vv)
    #                 kwargs[k] = vv
    #             else:
    #                 kwargs[k] = set_kwargs(v, title=k)
    #
    # return kwargs


def set_agent_kwargs(agent):
    class_name = type(agent).__name__
    type_dict = dtypes.get_dict_dtypes('agent', class_name=class_name)
    title = f'{class_name} args'
    kwargs = {}
    for p in list(type_dict.keys()):
        kwargs[p] = getattr(agent, p)
    new_kwargs = set_kwargs(kwargs, title, type_dict=type_dict)
    for p, v in new_kwargs.items():
        if p == 'unique_id':
            agent.set_id(v)
        else:
            setattr(agent, p, v)
    return agent


def object_menu(selected):
    object_list = ['', 'Larva', 'Food', 'Border']
    title = 'Select object type'
    layout = [
        [sg.Text(title)],
        [sg.Listbox(default_values=[selected], values=object_list, change_submits=False, size=(20, len(object_list)),
                    key='SELECTED_OBJECT',
                    enable_events=True)],
        [sg.Ok(), sg.Cancel()]]
    window = sg.Window(title, layout)
    while True:
        event, values = window.read()
        if event == 'Ok':
            sel = values['SELECTED_OBJECT'][0]
            break
        elif event in (None, 'Cancel'):
            sel = selected
            break
    window.close()
    return sel


class GraphList:
    def __init__(self, name, fig_dict={}):
        self.name = name
        self.fig_dict = fig_dict
        self.layout, self.list_key = self.init_layout(name, fig_dict)
        self.canvas, self.canvas_key = self.init_canvas(name)
        self.fig_agg = None
        self.draw_key = 'unreachable'

    def init_layout(self, name, fig_dict):
        list_key = f'{name}_GRAPH_LIST'
        values = list(fig_dict.keys())
        h = int(np.max([len(values), 5]))
        l = [
            [sg.Text('GRAPHS')],
            [sg.Listbox(values=values, change_submits=True, size=(20, h), key=list_key, auto_size_text=True)],
        ]

        return l, list_key

    def init_canvas(self, name):
        canvas_key = f'{name}_CANVAS'
        figure_w, figure_h = 800, 800
        canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key=canvas_key)]])
        # canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key=canvas_key)]])
        return canvas, canvas_key

    def draw_fig(self, window, fig):
        if self.fig_agg:
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_canvas(window[self.canvas_key].TKCanvas, fig)

    def update(self, window, fig_dict):
        self.fig_dict = fig_dict
        window.Element(self.list_key).Update(values=list(fig_dict.keys()))

    def evaluate(self, window, list_values):
        if len(list_values) > 0:
            choice = list_values[0]
            fig = self.fig_dict[choice]
            self.draw_fig(window, fig)

    def get_layout(self, as_col=True):
        if as_col:
            return sg.Col(self.layout)
        else:
            return self.layout


class ButtonGraphList(GraphList):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draw_key = f'{self.name}_DRAW_FIG'
        l = [sg.B('Graph args', **b6_kws, k=f'{self.name}_FIG_ARGS'),
             sg.B('Draw', **b6_kws, k=self.draw_key),
             sg.B('Save', **b6_kws, k=f'{self.name}_SAVE_FIG')]
        self.layout.append(l)
        self.fig, self.save_to, self.save_as = None, '', ''
        self.func, self.func_kwargs = None, {}

    def evaluate(self, window, list_values):
        if len(list_values) > 0:
            choice = list_values[0]
            if self.fig_dict[choice] != self.func:
                self.func = self.fig_dict[choice]
                self.func_kwargs = self.get_graph_kwargs(self.func)

    def get_graph_kwargs(self, func):
        signature = inspect.getfullargspec(func)
        kwargs = dict(zip(signature.args[-len(signature.defaults):], signature.defaults))
        for k in ['datasets', 'labels', 'save_to', 'save_as', 'return_fig', 'deb_dicts']:
            if k in kwargs.keys():
                del kwargs[k]
        return kwargs

    def generate(self, window, data):
        if self.func is not None and len(list(data.keys())) > 0:
            try:
                self.fig, self.save_to, self.save_as = self.func(datasets=list(data.values()), labels=list(data.keys()),
                                                                 return_fig=True, **self.func_kwargs)
                self.draw_fig(window, self.fig)
            except:
                print('Plot not available for these datasets')
                self.fig, self.save_to, self.save_as = None, '', ''

    def save_fig(self):
        if self.fig is not None:
            layout = [
                [sg.Text('Filename', size=(10, 1)), sg.In(default_text=self.save_as, k='SAVE_AS', size=(80, 1))],
                [sg.Text('Directory', size=(10, 1)), sg.In(self.save_to, k='SAVE_TO', size=(80, 1)),
                 sg.FolderBrowse(initial_folder=get_parent_dir(), key='SAVE_TO', change_submits=True)],
                [sg.Ok(), sg.Cancel()]]

            event, values = sg.Window('Save figure', layout).read(close=True)
            if event == 'Ok':
                save_as = values['SAVE_AS']
                save_to = values['SAVE_TO']
                filepath = os.path.join(save_to, save_as)
                self.fig.savefig(filepath, dpi=300)
                # save_canvas(window['GRAPH_CANVAS'].TKCanvas, filepath)
                # figure_agg.print_figure(filepath)
                print(f'Plot saved as {save_as}')

    def set_fig_args(self):
        self.func_kwargs = set_kwargs(self.func_kwargs, title='Graph arguments')


def delete_objects_window(selected):
    ids = [sel.unique_id for sel in selected]
    title = 'Delete objects?'
    layout = [
        [sg.Text(title)],
        [sg.Listbox(default_values=ids, values=ids, change_submits=False, size=(20, len(ids)), key='DELETE_OBJECTS',
                    enable_events=True)],
        [sg.Ok(), sg.Cancel()]]
    window = sg.Window(title, layout)
    while True:
        event, values = window.read()
        if event == 'Ok':
            res = True
            break
        elif event == 'Cancel':
            res = False
            break
    window.close()
    return res


class BtnInfo:
    def __init__(self, state=True):
        self.state = state  # Can have 3 states - True, False, None (toggle)


def draw_canvas(canvas, figure, side='top', fill='both', expand=1):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side=side, fill=fill, expand=expand)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


class DynamicGraph:
    def __init__(self, agent, pars=[], available_pars=None):
        # sg.change_look_and_feel('DarkBlue15')
        sg.theme('DarkBlue15')
        self.agent = agent
        if available_pars is None:
            available_pars = par_conf.get_runtime_pars()
        self.available_pars = available_pars
        self.pars = pars
        self.dt = self.agent.model.dt
        self.init_dur = 20
        self.window_size = (1550, 1000)
        self.canvas_size = (self.window_size[0] - 50, self.window_size[1] - 200)
        self.my_dpi = 96
        self.figsize = (int(self.canvas_size[0] / self.my_dpi), int(self.canvas_size[1] / self.my_dpi))

        Ncols = 4
        par_lists = [list(a) for a in np.array_split(self.available_pars, Ncols)]
        par_layout = [[sg.Text('Choose parameters')],
                      [sg.Col([*[[sg.CB(p, key=f'k_{p}', **t24_kws)] for p in par_lists[i]]]) for i in
                       range(Ncols)],
                      [sg.Button('Ok', **t8_kws), sg.Button('Cancel', **t8_kws)]
                      ]

        graph_layout = [
            # [sg.Text(f'{self.agent.unique_id} : {self.par}', size=(40, 1), justification='center', font='Helvetica 20')],
            [sg.Canvas(size=(1280, 1200), key='-CANVAS-')],
            [sg.Text('Time in seconds to display on screen')],
            [sg.Slider(range=(0.1, 60), default_value=self.init_dur, size=(40, 10), orientation='h',
                       key='-SLIDER-TIME-')],
            [sg.Button('Choose', **t8_kws)]
        ]
        layout = [[sg.Column(par_layout, key='-COL1-'), sg.Column(graph_layout, visible=False, key='-COL2-')]]
        self.window = sg.Window(f'{self.agent.unique_id} Dynamic Graph', layout, finalize=True, location=(0, 0),
                                size=self.window_size)
        self.canvas_elem = self.window.FindElement('-CANVAS-')
        self.canvas = self.canvas_elem.TKCanvas
        self.fig_agg = None

        self.update_pars()
        self.layout = 1

    def evaluate(self):
        event, values = self.window.read(timeout=0)
        if event is None:
            self.window.close()
            return False
        elif event == 'Choose':
            self.window[f'-COL2-'].update(visible=False)
            self.window[f'-COL1-'].update(visible=True)
            self.layout = 1
        elif event == 'Ok':
            self.window[f'-COL1-'].update(visible=False)
            self.window[f'-COL2-'].update(visible=True)
            self.pars = [p for p in self.available_pars if values[f'k_{p}']]
            self.update_pars()
            self.layout = 2
        elif event == 'Cancel':
            self.window[f'-COL1-'].update(visible=False)
            self.window[f'-COL2-'].update(visible=True)
            self.layout = 2

        if self.layout == 2 and self.Npars > 0:
            secs = values['-SLIDER-TIME-']
            Nticks = int(secs / self.dt)  # draw this many data points (on next line)
            t = self.agent.model.Nticks * self.dt
            trange = np.linspace(t - secs, t, Nticks)
            ys = self.update(Nticks)
            for ax, y in zip(self.axs, ys):
                ax.lines.pop(0)
                ax.plot(trange, y, color='black')
            self.axs[-1].set_xlim(np.min(trange), np.max(trange))
            self.fig_agg.draw()
        return True

    def update(self, Nticks):
        y_nan = np.ones(Nticks) * np.nan
        ys = []
        for p, v in self.yranges.items():
            self.yranges[p] = np.append(v, getattr(self.agent, p))
            dif = self.yranges[p].shape[0] - Nticks
            if dif >= 0:
                y = self.yranges[p][-Nticks:]
            else:
                y = y_nan
                y[-dif:] = self.yranges[p]
                # y = np.pad(self.yranges[p], (-dif, 0), constant_values=np.nan)
            ys.append(y)
        return ys

    def update_pars(self):
        to_return = ['par', 'symbol', 'exp_symbol', 'unit', 'lim', 'collect']
        self.pars, self.sim_symbols, self.exp_symbols, self.units, self.ylims, self.par_collects = par_conf.par_dict_lists(
            pars=self.pars, to_return=to_return)
        self.Npars = len(self.pars)
        self.yranges = {}

        self.fig, axs = plt.subplots(self.Npars, 1, figsize=self.figsize, dpi=self.my_dpi, sharex=True)
        if self.Npars > 1:
            self.axs = axs.ravel()
        else:
            self.axs = [axs]
        Nticks = int(self.init_dur / self.dt)
        for i, (ax, p, l, u, lim, p_col) in enumerate(
                zip(self.axs, self.pars, self.sim_symbols, self.units, self.ylims, self.par_collects)):
            if hasattr(self.agent, p_col):
                p0 = p_col
            else:
                p0 = p
            self.yranges[p0] = np.ones(Nticks) * np.nan
            ax.grid()
            ax.plot(range(Nticks), self.yranges[p0], color='black', label=l)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.legend(loc='upper right')
            ax.set_ylabel(u, fontsize=10)
            if lim is not None:
                ax.set_ylim(lim)
            ax.tick_params(axis='y', which='major', labelsize=10)
            if i == self.Npars - 1:
                ax.set_xlabel('time, $sec$')
                ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        self.fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.99, wspace=0.01, hspace=0.05)
        if self.fig_agg:
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_canvas(self.canvas, self.fig)


def fullNcap(conf_type):
    if conf_type == 'Env':
        full = 'environment'
        cap = 'ENV'
    elif conf_type == 'Batch':
        full = 'batch'
        cap = 'BATCH'
    elif conf_type == 'Model':
        full = 'model'
        cap = 'MODEL'
    elif conf_type == 'Exp':
        full = 'experiment'
        cap = 'EXP'
    return full, cap


def save_gui_conf(window, conf, conf_type):
    full, cap = fullNcap(conf_type)
    l = [
        named_list_layout(f'Store new {full}', f'{cap}_ID', list(loadConfDict(conf_type).keys()),
                          readonly=False, enable_events=False),
        [sg.Ok(), sg.Cancel()]]
    e, v = sg.Window(f'{full} configuration', l).read(close=True)
    if e == 'Ok':
        conf_id = v[f'{cap}_ID']
        saveConf(conf, conf_type, conf_id)
        window[f'{cap}_CONF'].update(values=list(loadConfDict(conf_type).keys()))
        window[f'{cap}_CONF'].update(value=conf_id)


def delete_gui_conf(window, values, conf_type):
    full, cap = fullNcap(conf_type)
    if values[f'{cap}_CONF'] != '':
        deleteConf(values[f'{cap}_CONF'], conf_type)
        window[f'{cap}_CONF'].update(values=list(loadConfDict(conf_type).keys()))
        window[f'{cap}_CONF'].update(value='')


def check_collapsibles(window, event, collapsibles):
    if event.startswith('OPEN SEC'):
        sec = event.split()[-1]
        if collapsibles[sec].state is not None:
            collapsibles[sec].state = not collapsibles[sec].state
            window[event].update(SYMBOL_DOWN if collapsibles[sec].state else SYMBOL_UP)
            window[f'SEC {sec}'].update(visible=collapsibles[sec].state)
    # return collapsibles


def check_toggles(window, event):
    if 'TOGGLE' in event:
        if window[event].metadata.state is not None:
            window[event].metadata.state = not window[event].metadata.state
            window[event].update(image_data=on_image if window[event].metadata.state else off_image)
