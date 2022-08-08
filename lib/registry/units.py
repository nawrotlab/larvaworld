from pint import UnitRegistry


ureg = UnitRegistry()
ureg.default_format = "~P"
ureg.setup_matplotlib(True)


