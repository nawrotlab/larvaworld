from pint import UnitRegistry


ureg = UnitRegistry()
ureg.default_format = "~L"
ureg.setup_matplotlib(True)
