
from lib.registry import reg
reg.init()
reg.init2()

def RvsSx4():

    from lib.sim.essay.essay_run import DoublePatch_Essay
    sufs=['foragers', 'navigators','feeders', 'locomotors']
    i=0
    for o in [True,False]:
        for f in [True,False]:
            E = DoublePatch_Essay(video=False, N=5, dur=3, olfactor=o,feeder=f,
                         id=f'Essay_DoublePatch_{sufs[i]}')
        #     # print(E.patch_env())
            ds = E.run()
            #figs, results = E.anal()
            i+=1
if __name__ == "__main__":
    pass
    # print(reg.DEF)
    # print(reg.DEF)
    # print(reg.CT)
    # print(reg.CT)
    # from functools import lru_cache


    # class Foo(object):
    #
    #     @property
    #     @lru_cache()
    #     def prop(self):
    #         print("called once")
    #         return 42
    #
    #
    # foo = Foo()
    # print(foo.prop)
    # print(foo.prop)
    # print(reg.datapath('step','fff'))
    # RvsSx4()