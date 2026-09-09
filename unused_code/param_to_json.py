import larvaworld
from larvaworld.lib import reg, param
from larvaworld.lib.util import AttrDict
from larvaworld.lib.param.custom import restore_param_class, _deserialize_param_value

# A=reg.LarvaworldParam.set_vparam(param.Number(default=1.0, bounds=(0.0, 10.0), doc="Custom v parameter"))()
# print(A.param.objects()["v"].__getstate__()["default"])
# print(A.param.objects()["v"])
# raise
P = reg.par.get_param("t")
print(reg.LarvaworldParam.param_keys())
print(P.param_keys())
# raise
tmp_path = "t_param.json"

P.save_config(tmp_path)
config = AttrDict.load(tmp_path)
# xx = {k: _deserialize_param_value(v) for k, v in config.items() if k in reg.LarvaworldParam.param_keys()}
PP = reg.LarvaworldParam.from_config(config)

print(P.param.objects())
print(PP.param.objects())
# value=float
# x=f"{value.__module__}.{value.__qualname__}"
# x=xx["dtype"]
# print(x)
# print(restore_param_class(x))
# print(type(restore_param_class(x)))
print(P.dtype, PP.dtype)
print(
    P.param.objects()["v"].__getstate__()["default"],
    PP.param.objects()["v"].__getstate__()["default"],
)
