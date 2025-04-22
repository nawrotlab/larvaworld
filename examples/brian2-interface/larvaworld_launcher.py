import larvaworld.lib.reg as reg
import larvaworld.lib.sim as sim
import larvaworld.lib.model.modules as mods
from larvaworld.lib.model import BrainModuleDB

from larvaworld.lib.model.modules.remote_brian_interface import (
    RemoteBrianModelInterface,
)
from sensors import LocalOlfactor, CustomBehaviorModule

# overwrite mode 'osn' to use our custom LocalOlfactor class
BrainModuleDB.BrainModuleModes.olfactor.osn = LocalOlfactor

# load predefined experiment
expID = "chemorbit_OSN"
exp_conf = reg.conf.Exp.getID(expID)
print(
    f"default odor intensity: {exp_conf.env_params.food_params.source_units.Source.odor}"
)
# exp_conf.env_params.food_params.source_units.Source.odor.intensity = 0
exp_conf.env_params.food_params.source_units.Source.odor.spread = 0.01
print(f"new odor intensity: {exp_conf.env_params.food_params.source_units.Source.odor}")


larva_group = exp_conf.larva_groups
print(f"larva_groups: {larva_group.keylist}")
print(larva_group)
larva_group_id = larva_group.keylist[
    0
]  # get the "name" of the larva group - this defaults to the model_id of this group (if none is provided)
# larva_group[larva_group_id].model is the SAFE way to obtain the model_id

# exp_conf.larva_groups[larva_group_id].model = 'Levy_navigator' # change model config to Levy_navigator
print(
    larva_group_id, larva_group[larva_group_id].model
)  # these are both the same thing if no custom larva_group name is provided

mm = reg.conf.Model.getID(larva_group[larva_group_id].model)
# set olfactor mode to osn to use our custom LocalOlfactor implementation
mm.brain.olfactor.mode = "osn"
# mm.brain.turner.amp = 10.0
# mm.brain.crawler.amp = 0.75
# print(f"larva model config: {mm}")

run_id = RemoteBrianModelInterface.getRandomModelId()
print("Run_id: {} videoFile: videos/OSN_larva_{}".format(run_id, run_id))

# This runs the simulation
# You can omit the screen_kws to not render display
erun = sim.ExpRun(
    experiment=expID,
    # id='simId',
    modelIDs=["navigator", "OSNnavigator"],
    screen_kws={
        "vis_mode": "video",
        "show_display": True,
        "save_video": True,
        "fps": 20,
        "video_file": "OSN_larva_{}".format(run_id),
        "media_dir": "videos/",
    },
    N=2,
    duration=0.5,
)
erun.simulate()
erun.analyze()
