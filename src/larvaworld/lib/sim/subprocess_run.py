import subprocess
import argparse

from larvaworld.lib import reg, aux, sim
from larvaworld.lib.process.dataset import LarvaDataset


class Exec:
    def __init__(self, mode, conf, run_externally=True, progressbar=None, w_progressbar=None, **kwargs):
        self.run_externally = run_externally
        self.mode = mode
        self.conf = conf
        self.progressbar = progressbar
        self.w_progressbar = w_progressbar
        self.type = self.conf['batch_type'] if mode == 'batch' else self.conf['experiment']
        self.done = False


    def terminate(self):
        if self.process is not None:
            self.process.terminate()
            self.process.kill()

    def run(self, **kwargs):
        f0, f1 = f'{reg.ROOT_DIR}/lib/sim/exec_conf.txt', f'{reg.ROOT_DIR}/lib/sim/exec_run.py'
        if self.run_externally:
            aux.save_dict(self.conf, f0)
            self.process = subprocess.Popen(['python', f1, self.mode, f0], **kwargs)
        else:
            res = self.exec_run()
            self.results = self.retrieve(res)
            self.done = True

    def check(self):
        if not self.done:
            if self.run_externally:
                if self.process.poll() is not None:
                    self.results = self.retrieve()
                    self.done = True
                    return True
            return False
        else:
            return True

    def retrieve(self, res=None):
        if self.mode == 'batch':
            if res is None and self.run_externally:
                args = {'batch_type': self.type, 'id': self.conf['id']}
                res = aux.retrieve_results(**args)
            return res
        elif self.mode == 'sim':
            id = self.conf['id']
            if res is None and self.run_externally:
                dir0 = f"{reg.SIM_DIR}/single_runs/{self.conf['sim_params']['path']}/{id}"
                res = [LarvaDataset(f'{dir0}/{id}.{gID}') for gID in self.conf['larva_groups'].keys()]

            if res is not None:
                # TODO sim analysis independent from SingleRun class. Currently exec does not run analysis for "sim" mode
                # fig_dict, results = sim_analysis(res, self.type)
                fig_dict, results = None, None
                entry = {id: {'dataset': res, 'figs': fig_dict}}
            else:
                entry, fig_dict = None, None
            return entry, fig_dict

    def exec_run(self):

        if self.mode == 'sim':
            self.process = sim.ExpRun(parameters=self.conf)
            res = self.process.simulate()
        elif self.mode == 'batch':
            self.process = None
            k = sim.BatchRun(**self.conf)
            res=k.simulate()

        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run given batch-exec/simulation")
    parser.add_argument('mode', choices=['sim', 'batch'],
                        help='Whether we are running a single simulation or a batch-exec')
    parser.add_argument('conf_file', type=str, help='The configuration file of the batch-exec/simulation')
    args = parser.parse_args()
    conf = aux.load_dict(args.conf_file)
    k = Exec(args.mode, conf)
    k.exec_run()

