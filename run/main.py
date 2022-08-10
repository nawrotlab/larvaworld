
from lib.registry import reg
reg.init()

def RvsSx4():
    from lib.conf.stored.essay_conf import DoublePatch_Essay
    sufs=['foragers', 'navigators','feeders', 'locomotors']
    i=0
    for o in [True,False]:
        for f in [True,False]:
            E = DoublePatch_Essay(video=False, N=10, dur=5, olfactor=o,feeder=f, essay_id=f'RvsS_{sufs[i]}')
        #     # print(E.patch_env())
            ds = E.run()
            figs, results = E.anal()
            i+=1
if __name__ == "__main__":
    pass
