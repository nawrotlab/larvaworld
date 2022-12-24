from pint import UnitRegistry


units = UnitRegistry()
units.default_format = "~P"
units.setup_matplotlib(True)


