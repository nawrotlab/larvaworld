import random

import PySimpleGUI as sg
import numpy as np

import lib.aux.functions as fun
from lib.conf.dtype_dicts import border_dtypes, food_dtypes, odor_dtypes, arena_dtypes, distro_dtypes, null_distro, \
    odor_null_distro, food_null_distro
from lib.gui.gui_lib import CollapsibleDict, b_kws, t_kws, check_collapsibles, check_toggles, \
    retrieve_dict, t5_kws, t2_kws

"""
    Demo - Drawing and moving demo

    This demo shows how to use a Graph Element to (optionally) display an image and then use the
    mouse to "drag" and draw rectangles and circles.
"""

W, H = 800, 800


def update_window_distro(window, name, p1, p2):
    scale = tuple(np.abs(np.array(p2) - np.array(p1)))
    window[f'{name}_DISTRO_scale'].update(value=scale)
    window[f'{name}_DISTRO_loc'].update(value=p1)


def draw_shape(graph, p1, p2, shape, **kwargs):
    if p2 == p1:
        return None
    pp1, pp2 = np.array(p1), np.array(p2)
    dpp = np.abs(pp2 - pp1)
    if shape == 'rect':
        p1 = tuple(pp1 - dpp / 2)
        p2 = tuple(pp1 + dpp / 2)
        fig = graph.draw_rectangle(p1, p2, **kwargs)
    elif shape == 'circle':
        fig = graph.draw_circle(p1, dpp[0], **kwargs)
    else:
        fig = None
    return fig


def popup_color_chooser(look_and_feel=None):
    """

    :return: Any(str, None) Returns hex string of color chosen or None if nothing was chosen
    """
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


def color_pick_layout(name, color):
    return [sg.T('', **t5_kws), sg.T('color', **t5_kws),
            sg.In(default_text=f'{color}', k=f'{name}_color', **t_kws),
            sg.B('Pick', k=f'PICK {name}_color', **b_kws)]


def add_agent_layout(name0, color, collapsibles, dtype_name=None, basic=True):
    if dtype_name is None:
        dtype_name = name0
    name = name0.upper()

    collapsibles[f'{name}_DISTRO'] = CollapsibleDict(f'{name}_DISTRO', False, dict=null_distro(dtype_name, basic=basic),
                                                     type_dict=distro_dtypes(dtype_name, basic=basic),
                                                     toggle=False, disabled=True)

    collapsibles[f'{name}_ODOR'] = CollapsibleDict(f'{name}_ODOR', False, dict=odor_null_distro, type_dict=odor_dtypes,
                                                   toggle=False)
    l = [[sg.R(f'Add {name0}', 1, k=name, enable_events=True)],
         [sg.T('', **t2_kws),
          sg.R('single id', 2, disabled=True, k=f'{name}_single', enable_events=True, **t5_kws),
          sg.In(f'{name}_0', k=f'{name}_id', **t_kws)],
         [sg.T('', **t2_kws), sg.R('group id', 2, disabled=True, k=f'{name}_group', enable_events=True, **t5_kws),
          sg.In(k=f'{name}_group_id', **t_kws)],
         color_pick_layout(name, color),
         [sg.T('', **t5_kws), *collapsibles[f'{name}_DISTRO'].get_section()],
         [sg.T('', **t5_kws), *collapsibles[f'{name}_ODOR'].get_section()]]
    return l, collapsibles


def draw_arena(graph, arena_pars):
    graph.erase()
    shape = arena_pars['arena_shape']
    X, Y = arena_pars['arena_xdim'], arena_pars['arena_ydim']
    if shape == 'circular' and X is not None:
        arena = graph.draw_circle((int(W / 2), int(H / 2)), int(W / 2), fill_color='white', line_color='black',
                                  line_width=5)
        s = W / X
    elif shape == 'rectangular' and not None in (X, Y):
        if X >= Y:
            dif = (X - Y) / X
            arena = graph.draw_rectangle((0, int(H * dif / 2)), (W, H - int(H * dif / 2)), fill_color='white',
                                         line_color='black', line_width=5)
            s = W / X
        else:
            dif = (Y - X) / Y
            arena = graph.draw_rectangle((int(W * dif / 2), 0), (W - int(W * dif / 2), H), fill_color='white',
                                         line_color='black')
            s = H / Y
    return s, arena
    # pass


def scale_xy(xy, s):
    return (xy[0] - W / 2) / s, (xy[1] - H / 2) / s


def unscale_xy(xy, s):
    return xy[0] * s + W / 2, xy[1] * s + H / 2


def out_of_bounds(xy, arena_pars):
    shape = arena_pars['arena_shape']
    X, Y = arena_pars['arena_xdim'], arena_pars['arena_ydim']
    x, y = xy
    if shape == 'circular':
        return np.sqrt(x ** 2 + y ** 2) > X / 2
    elif shape == 'rectangular':
        return not (-X / 2 < x < X / 2 and -Y / 2 < y < Y / 2)


def delete_prior(prior_rect, graph):
    # print('xx')
    if type(prior_rect) == list:
        for pr in prior_rect:
            graph.delete_figure(pr)
    else:
        graph.delete_figure(prior_rect)


def inspect_distro(default_color, mode, shape, N, loc, scale, graph, s, id=None, item='LARVA', **kwargs):
    Ps = fun.generate_xy_distro(mode, shape, N, loc=unscale_xy(loc, s), scale=np.array(scale) * s)
    group_figs = []
    for i, P0 in enumerate(Ps):
        if item == 'SOURCE':
            temp = draw_source(P0, default_color, graph, s, **kwargs)
        elif item == 'LARVA':
            temp = draw_larva(P0, default_color, graph, s, **kwargs)
        group_figs.append(temp)
    return group_figs


def draw_source(P0, color, graph, s, amount, radius, **kwargs):
    fill_color = color if amount > 0 else None
    temp = graph.draw_circle(P0, radius * s, line_width=3, line_color=color, fill_color=fill_color)
    return temp


def draw_larva(P0, color, graph, s, orientation_range, **kwargs):
    points = np.array([[0.9, 0.1], [0.05, 0.1]])
    xy0 = fun.body(points)
    a1, a2 = orientation_range
    a1, a2 = np.deg2rad(a1), np.deg2rad(a2)
    xy0 = fun.rotate_multiple_points(xy0, random.uniform(a1, a2), origin=[0, 0])
    # print(xy0)
    xy0 /= 250
    # print(xy0)
    xy0 *= s
    # print(xy0)
    xy0 += np.array(P0)
    # print(xy0)
    temp = graph.draw_polygon(xy0, line_width=3, line_color=color, fill_color=color)
    return temp


def check_abort(name, w, v, units, groups):
    o = name
    info = w['info']
    abort = True
    odor_on = w[f'TOGGLE_{o}_ODOR'].metadata.state

    if not odor_on:
        w[f'{o}_ODOR_odor_id'].update(value=None)
        w[f'{o}_ODOR_odor_intensity'].update(value=0.0)

    if o == 'SOURCE':
        food_on = w[f'TOGGLE_{o}_FOOD'].metadata.state
        if not odor_on and not food_on:
            info.update(value=f"Assign food and/or odor to the drawn source")
            return True
        else:
            if not food_on and v[f'{o}_FOOD_amount'] != 0.0:
                w[f'{o}_FOOD_amount'].update(value=0.0)
                info.update(value=f"Source food amount set to 0")
            elif v[f'{o}_FOOD_amount'] == 0.0:
                w[f'{o}_FOOD_amount'].update(value=10 ** -3)
                info.update(value=f"Source food amount set to default")
    if v[f'{o}_group_id'] == '' and v[f'{o}_id'] == '':
        info.update(value=f"Both {o.lower()} single id and group id are empty")
    elif not v[f'{o}_group'] and not v[f'{o}_single']:
        info.update(value=f"Select to add a single or a group of {o.lower()}s")
    elif v[f'{o}_single'] and (
            v[f'{o}_id'] in list(units.keys()) or v[f'{o}_id'] == ''):
        info.update(value=f"{o.lower()} id {v[f'{o}_id']} already exists or is empty")
    elif odor_on and v[f'{o}_ODOR_odor_id'] == '':
        info.update(value=f"Default odor id automatically assigned to the odor")
        id = v[f'{o}_group_id'] if v[f'{o}_group_id'] != '' else v[f'{o}_id']
        w[f'{o}_ODOR_odor_id'].update(value=f'{id}_odor')
    elif odor_on and not float(v[f'{o}_ODOR_odor_intensity']) > 0:
        info.update(value=f"Assign positive odor intensity to the drawn odor source")
    elif odor_on and (
            v[f'{o}_ODOR_odor_spread'] == '' or not float(v[f'{o}_ODOR_odor_spread']) > 0):
        info.update(value=f"Assign positive spread to the odor")
    elif v[f'{o}_group'] and (
            v[f'{o}_group_id'] in list(groups.keys()) or v[f'{o}_group_id'] == ''):
        info.update(value=f"{o.lower()} group id {v[f'{o}_group_id']} already exists or is empty")
    elif v[f'{o}_group'] and v[f'{o}_DISTRO_mode'] in ['', None]:
        info.update(value=f"Define a distribution mode")
    elif v[f'{o}_group'] and v[f'{o}_DISTRO_shape'] in ['', None]:
        info.update(value=f"Define a distribution shape")
    elif v[f'{o}_group'] and not int(v[f'{o}_DISTRO_N']) > 0:
        info.update(value=f"Assign a positive integer number of items for the distribution")
    else:
        abort = False
    # print(abort, o)
    return abort


def draw_env(env=None):
    sg.theme('LightGreen')
    # sg.theme('Dark Blue 3')
    collapsibles = {}
    if env is None:
        env = {'border_list': {},
               'arena_params': {'arena_xdim': 0.1,
                                'arena_ydim': 0.1,
                                'arena_shape': 'circular'},
               'food_params': {'source_units': {}, 'source_groups': {}, 'food_grid': None},
               'larva_params': {}
               }

    borders = env['border_list']
    arena_pars = env['arena_params']
    source_units = env['food_params']['source_units']
    source_groups = env['food_params']['source_groups']
    larva_groups = env['larva_params']
    larva_units = {}
    borders_f, source_units_f, source_groups_f, larva_units_f, larva_groups_f = {}, {}, {}, {}, {}
    inspect_figs = {}
    sample_fig, sample_pars = None, {}

    collapsibles['ARENA'] = CollapsibleDict('ARENA', True, dict=arena_pars, type_dict=arena_dtypes)

    source_l, collapsibles = add_agent_layout('Source', 'green', collapsibles, dtype_name='Food')
    larva_l, collapsibles = add_agent_layout('Larva', 'black', collapsibles)

    collapsibles['SOURCE_FOOD'] = CollapsibleDict('SOURCE_FOOD', False, dict=food_null_distro, type_dict=food_dtypes,
                                                  toggle=False)

    s = None
    arena = None

    col2 = [

        *larva_l,

        *source_l,

        [sg.T('', **t5_kws), *collapsibles['SOURCE_FOOD'].get_section()],

        [sg.T('', **t5_kws), sg.T('shape', **t5_kws),
         sg.Combo(['rect', 'circle'], default_value='circle', k='SOURCE_shape', enable_events=True, readonly=True,
                  **t_kws)],

        [sg.R('Add Border', 1, k='BORDER', enable_events=True)],
        [sg.T('', **t5_kws), sg.T('id', **t5_kws),
         sg.In(f'BORDER_{len(borders.keys())}', k='BORDER_id', **t_kws)],
        [sg.T('', **t5_kws), sg.T('width', **t5_kws), sg.In(0.001, k='BORDER_width', **t_kws)],
        color_pick_layout('BORDER', 'black'),

        [sg.R('Erase item', 1, k='-ERASE-', enable_events=True)],
        [sg.R('Move item', 1, True, k='-MOVE-', enable_events=True)],
        [sg.R('Inspect item', 1, True, k='-INSPECT-', enable_events=True)],
    ]

    col1 = [
        collapsibles['ARENA'].get_section(), [sg.B('Reset arena', k='RESET_ARENA', **b_kws)],
        [sg.Graph(
            canvas_size=(W, H),
            graph_bottom_left=(0, 0),
            graph_top_right=(W, H),
            key="-GRAPH-",
            change_submits=True,  # mouse click events
            background_color='black',
            drag_submits=True)],
        [sg.T('Instructions : ', k='info', size=(60, 3))],
        [sg.B('Ok', **b_kws), sg.B('Cancel', **b_kws)]
    ]
    layout = [[sg.Col(col1), sg.Col(col2)]]

    w = sg.Window("Drawing and Moving Stuff Around", layout, finalize=True)

    graph = w["-GRAPH-"]  # type: sg.Graph

    dragging, current = False, {}
    start_point = end_point = prior_rect = None

    graph.bind('<Button-3>', '+RIGHT+')
    while True:
        e, v = w.read()
        info = w["info"]
        if e in [None, 'Cancel']:
            break
        elif e == 'Ok':
            env['arena_params'] = collapsibles['ARENA'].get_dict(v, w)
            env['border_list'] = borders
            env['food_params']['source_units'] = source_units
            env['food_params']['source_groups'] = source_groups
            env['larva_params'] = larva_groups
            break  # exit
        check_collapsibles(w, e, collapsibles)
        check_toggles(w, e)
        if e == 'RESET_ARENA':
            info.update(value='Arena has been reset. All items erased.')
            s, arena = draw_arena(graph, collapsibles['ARENA'].get_dict(v, w))
            borders, source_units, source_groups, larva_units, larva_groups = {}, {}, {}, {}, {}
            borders_f, source_units_f, source_groups_f, larva_units_f, larva_groups_f = {}, {}, {}, {}, {}

            for ii in ['BORDER', 'SOURCE', 'LARVA']:
                w[f'{ii}_id'].update(value=f'{ii}_0')

        if arena is None:
            continue
        if e == '-MOVE-':
            graph.Widget.config(cursor='fleur')
            # graph.set_cursor(cursor='fleur')          # not yet released method... coming soon!
        elif not e.startswith('-GRAPH-'):
            # graph.set_cursor(cursor='left_ptr')       # not yet released method... coming soon!
            graph.Widget.config(cursor='left_ptr')
        if e.startswith('PICK'):
            target = e.split()[-1]
            choice = popup_color_chooser('Dark Blue 3')
            w[target].update(choice)
        if e == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = v["-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
                drag_figures = graph.get_figures_at_location((x, y))
                lastxy = x, y
            else:
                end_point = (x, y)
            if prior_rect:
                delete_prior(prior_rect, graph)
            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = x, y
            if None not in (start_point, end_point):
                if v['-MOVE-']:
                    delta_X, delta_Y = delta_x / s, delta_y / s
                    for fig in drag_figures:
                        if fig != arena:
                            for dic, f_dic in zip([borders, source_units, source_groups, larva_units, larva_groups],
                                                  [borders_f, source_units_f, source_groups_f, larva_units_f,
                                                   larva_groups_f]):
                                if fig in list(f_dic.keys()):
                                    w["info"].update(value=f"Item {f_dic[fig]} moved by ({delta_X}, {delta_Y})")
                                    if dic == source_units:
                                        X0, Y0 = dic[f_dic[fig]]['pos']
                                        dic[f_dic[fig]]['pos'] = (X0 + delta_X, Y0 + delta_Y)
                                        print(dic[f_dic[fig]]['pos'])
                                    elif dic in [source_groups, larva_groups]:
                                        X0, Y0 = dic[f_dic[fig]]['loc']
                                        dic[f_dic[fig]]['loc'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif dic == borders:
                                        dic[f_dic[fig]]['points'] = [(X0 + delta_X, Y0 + delta_Y) for X0, Y0 in
                                                                     dic[f_dic[fig]]['points']]
                            graph.move_figure(fig, delta_x, delta_y)
                            graph.update()
                elif v['-ERASE-']:
                    for figure in drag_figures:
                        if figure != arena:
                            # print(figure)
                            for dic, f_dic in zip([borders, source_units, source_groups, larva_groups],
                                                  [borders_f, source_units_f, source_groups_f, larva_groups_f]):
                                if figure in list(f_dic.keys()):
                                    w["info"].update(value=f"Item {f_dic[figure]} erased")
                                    dic.pop(f_dic[figure])
                                    f_dic.pop(figure)
                            graph.delete_figure(figure)
                elif v['-INSPECT-']:
                    for figure in drag_figures:
                        if figure != arena:
                            for dic, f_dic in zip([borders, source_units, source_groups, larva_groups],
                                                  [borders_f, source_units_f, source_groups_f, larva_groups_f]):

                                if figure in list(f_dic.keys()):
                                    if f_dic[figure] in list(inspect_figs.keys()):
                                        for f in inspect_figs[f_dic[figure]]:
                                            graph.delete_figure(f)
                                        inspect_figs.pop(f_dic[figure])
                                    else:
                                        w["info"].update(value=f"Inspecting item {f_dic[figure]} ")
                                        if dic in [source_groups, larva_groups]:
                                            inspect_figs[f_dic[figure]] = inspect_distro(id=f_dic[figure], s=s,
                                                                                         graph=graph,
                                                                                         **dic[f_dic[figure]])

                elif v['SOURCE'] or v['BORDER'] or v['LARVA']:
                    P1, P2 = scale_xy(start_point, s), scale_xy(end_point, s)
                    if any([out_of_bounds(P, collapsibles['ARENA'].get_dict(v, w)) for P in [P1, P2]]):
                        current = {}
                    else:
                        if v['SOURCE'] and not check_abort('SOURCE', w, v, source_units, source_groups):
                            o = 'SOURCE'
                            color = v[f'{o}_color']
                            if v['SOURCE_single'] or (v['SOURCE_group'] and sample_fig is None):

                                c = {'fill_color': color if w['TOGGLE_SOURCE_FOOD'].metadata.state else None,
                                     'line_color': color,
                                     'line_width': 5,
                                     }
                                prior_rect = draw_shape(graph, shape=v[f'{o}_shape'], p1=start_point,
                                                        p2=end_point, **c)

                                w['SOURCE_FOOD_radius'].update(value=np.abs(end_point[0] - start_point[0]) / s)
                                food_pars = collapsibles['SOURCE_FOOD'].get_dict(v, w)
                                odor_pars = collapsibles['SOURCE_ODOR'].get_dict(v, w)
                                sample_pars = {'default_color': color,
                                               **food_pars,
                                               **odor_pars,
                                               }

                                if v['SOURCE_single']:

                                    current = {v['SOURCE_id']: {
                                        'group': v['SOURCE_group_id'],
                                        'pos': P1,
                                        **sample_pars
                                    }}
                                    sample_fig, sample_pars = None, {}
                                else:
                                    info.update(value=f"Draw a sample item for the distribution")
                            elif v[f'{o}_group']:
                                update_window_distro(w, o, P1, P2)
                                distro_pars = collapsibles['SOURCE_DISTRO'].get_dict(v, w)
                                current = {v['SOURCE_group_id']: {
                                    **distro_pars,
                                    **sample_pars
                                }}
                                c = {'fill_color': None,
                                     'line_color': color,
                                     'line_width': 5,
                                     }
                                prior_rect = draw_shape(graph, shape=v[f'{o}_DISTRO_shape'], p1=start_point,
                                                        p2=end_point, **c)
                        elif v['LARVA'] and not check_abort('LARVA', w, v, larva_units, larva_groups):
                            o = 'LARVA'
                            color = v[f'{o}_color']
                            odor_pars = collapsibles[f'{o}_ODOR'].get_dict(v, w)
                            sample_larva_pars = {'default_color': color,
                                                 **odor_pars,
                                                 }
                            if v[f'{o}_group']:
                                update_window_distro(w, o, P1, P2)
                                distro_pars = collapsibles[f'{o}_DISTRO'].get_dict(v, w)
                                current = {v[f'{o}_group_id']: {
                                    **distro_pars,
                                    **sample_larva_pars
                                }}
                                c = {'fill_color': None,
                                     'line_color': color,
                                     'line_width': 5,
                                     }
                                prior_rect = draw_shape(graph, shape=v[f'{o}_DISTRO_shape'], p1=start_point,
                                                        p2=end_point, **c)

                        elif v['BORDER']:
                            id = v['BORDER_id']
                            if id in list(borders.keys()) or id == '':
                                info.update(value=f"Border id {id} already exists or is empty")
                            else:
                                dic = {'unique_id': id,
                                       'default_color': v['BORDER_color'],
                                       'width': v['BORDER_width'],
                                       'points': [P1, P2]}
                                current = fun.agent_list2dict([retrieve_dict(dic, border_dtypes)])

                                prior_rect = graph.draw_line(start_point, end_point, color=v['BORDER_color'],
                                                             width=int(float(v['BORDER_width']) * s))



        elif e.endswith('+UP'):  # The drawing has ended because mouse up
            if v['BORDER'] and current != {}:
                info.update(value=f"Border {v['BORDER_id']} placed from {P1} to {P2}")
                borders_f[prior_rect] = id
                borders.update(current)
                w['BORDER_id'].update(value=f'BORDER_{len(borders.keys())}')
            elif v['SOURCE']:
                if v['SOURCE_single'] and current != {}:
                    info.update(value=f"Source {v['SOURCE_id']} placed at {P1}")
                    source_units_f[prior_rect] = v['SOURCE_id']
                    source_units.update(current)
                    w['SOURCE_id'].update(value=f'SOURCE_{len(source_units.keys())}')
                    w['SOURCE_ODOR_odor_id'].update(value='')
                elif v['SOURCE_group'] and sample_pars != {}:
                    if current == {}:
                        info.update(value=f"Sample item for source group {v['SOURCE_group_id']} detected." \
                                          "Now draw the distribution's space")

                        sample_fig = prior_rect
                    else:
                        id = v['SOURCE_group_id']
                        info.update(value=f"Source group {id} placed at {P1}")
                        source_groups_f[prior_rect] = id
                        source_groups.update(current)
                        w['SOURCE_group_id'].update(value=f'SOURCE_GROUP_{len(source_groups.keys())}')
                        w['SOURCE_ODOR_odor_id'].update(value='')
                        inspect_distro(id=id, **source_groups[id], graph=graph, s=s, item='SOURCE')
                        delete_prior(sample_fig, graph)
                        sample_fig, sample_pars = None, {}
            elif v['LARVA'] and current != {}:
                o = 'LARVA'
                units, groups = larva_units, larva_groups
                units_f, groups_f = larva_units_f, larva_groups_f
                if v[f'{o}_single']:
                    pass
                elif v[f'{o}_group']:
                    id = v[f'{o}_group_id']
                    info.update(value=f"{o} group {id} placed at {P1}")
                    groups_f[prior_rect] = id
                    groups.update(current)
                    w[f'{o}_group_id'].update(value=f'{o}_GROUP_{len(groups.keys())}')
                    w[f'{o}_ODOR_odor_id'].update(value='')
                    inspect_distro(id=id, **groups[id], graph=graph, s=s, item=o)
                    # delete_prior(sample_fig, graph)
                    sample_larva_pars = {}
            else:
                delete_prior(prior_rect, graph)

            dragging, current = False, {}
            start_point = end_point = prior_rect = None

        for o in ['SOURCE', 'LARVA']:
            w[f'{o}_single'].update(disabled=not v[o])
            w[f'{o}_group'].update(disabled=not v[o])
            collapsibles[f'{o}_DISTRO'].disable(w) if not v[f'{o}_group'] else collapsibles[f'{o}_DISTRO'].enable(w)
            if v[f'{o}_group']:
                w[f'{o}_id'].update(value='')
            # print(o, v[f'{o}_group'], int(v[f'{o}_DISTRO_N']))
            # elif v['SOURCE_id'] == '':
            #     w['SOURCE_id'].update(value=f'SOURCE_{len(source_units.keys())}')
        # print(list(borders.keys()))
    #
    w.close()
    return env


if __name__ == '__main__':
    draw_env()
