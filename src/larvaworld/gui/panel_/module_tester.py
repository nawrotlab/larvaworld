import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
from panel.template import DarkTheme
from panel.widgets import EditableRangeSlider

# from panel.template import DarkTheme
# from my_template import DarkTheme

from larvaworld.lib import reg, aux, model
from larvaworld.lib.model.modules import NeuralOscillator

module_class=NeuralOscillator
module_attrs=['input', 'activation', 'output']
module_description='Neural lateral oscillator'


def new_class(cls, **kwargs):
    "Creates a new class which overrides parameter defaults."
    return type(type(cls).__name__, (cls,), kwargs)

module_conf=pn.Param(module_class.param,
                     expand_button = True,
                    # default_layout=new_class(pn.GridBox, ncols=2),
                    default_precedence=3,
                    show_name=False,
                     widgets={
        "base_activation": {
                "type": pn.widgets.EditableFloatSlider},
        "activation_range": {
                "type": pn.widgets.EditableRangeSlider},
        "w_ee": {"type": pn.widgets.FloatInput},
        "w_ce": pn.widgets.FloatInput,
        "w_ec": pn.widgets.FloatInput,
        "w_cc": pn.widgets.FloatInput,
        "n": pn.widgets.FloatInput,
        "m": pn.widgets.IntInput,
        "tau": pn.widgets.FloatInput,
        "dt": pn.widgets.FloatInput,

                     })

# module_conf.widgets['base_activation']= pn.widgets.EditableFloatSlider
# module_conf.widgets['activation_range']= pn.widgets.EditableRangeSlider



# Data and Widgets
N=100
trange = np.arange(N)
A_in = pn.widgets.FloatSlider(name="A_in", start=-1, end=1, value=0)
#phase = pn.widgets.FloatSlider(name="Phase", start=0, end=np.pi)

p1=pn.Column(
    pn.pane.Markdown(f"### {module_description}"),
    # module_conf,
    # pn.Column(
    module_conf.widget('base_activation'),
    module_conf.widget('activation_range'),
    sizing_mode='stretch_width'
)
        # act_range
        # ),
p2=pn.GridBox(
module_conf.widget('tau'),
    module_conf.widget('dt'),
    module_conf.widget('w_ee'),
    module_conf.widget('w_ce'),
    module_conf.widget('w_ec'),
    module_conf.widget('w_cc'),
    module_conf.widget('m'),
    module_conf.widget('n'),


    ncols=2)
    # pn.pane.Markdown("#### Power Curve"),
    # power_curve_view,


# Interactive data pipeline
def module_tester(A_in):
    # kws=dict(module_class.param.get_param_values())
    # kws.pop('name')
    M = module_class()
    df=pd.DataFrame(columns=module_attrs, index=trange)
    for i in range(N):
        M.step(A_in=A_in)
        df.loc[i]={k:getattr(M,k) for k in module_attrs}
    return df

plot = hvplot.bind(module_tester,A_in).interactive()

template = pn.template.MaterialTemplate(title='Material Dark', theme=DarkTheme)

template.sidebar.append(A_in)
template.sidebar.append(p1)
template.sidebar.append(p2)

template.main.append(
    pn.Card(plot.hvplot(min_height=400).output(), title=module_description)
)
template.servable();

# Run from terminal with : panel serve module_tester.py --show --autoreload