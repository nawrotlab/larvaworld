import subprocess
import argparse

from lib import reg, aux
from lib.process.larva_dataset import LarvaDataset
from lib.sim.batch_aux import retrieve_results
from lib.sim.single_run import SingleRun
from lib.sim.batch import BatchRun


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
        # f0, f1 = preg.path_dict["EXECONF"],preg.path_dict["EXEC"]
        f0, f1 = reg.Path.EXECONF, reg.Path.EXEC
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
                args = {'batch_type': self.type, 'batch_id': self.conf['batch_id']}
                res = retrieve_results(**args)
            return res
        elif self.mode == 'sim':
            sim_id = self.conf['sim_params']['sim_ID']
            if res is None and self.run_externally:
                dir0 = f"{reg.Path.SIM}/{self.conf['sim_params']['path']}/{sim_id}"
                res = [LarvaDataset(f'{dir0}/{sim_id}.{gID}') for gID in self.conf['larva_groups'].keys()]

            if res is not None:
                # TODO sim analysis independent from SingleRun class. Currently exec does not run analysis for "sim" mode
                # fig_dict, results = sim_analysis(res, self.type)
                fig_dict, results = None, None
                entry = {sim_id: {'dataset': res, 'figs': fig_dict}}
            else:
                entry, fig_dict = None, None
            return entry, fig_dict

    def exec_run(self):

        if self.mode == 'sim':
            self.process = SingleRun(**self.conf, progress_bar=self.w_progressbar)
            res = self.process.run()
        elif self.mode == 'batch':
            self.process = None
            k = BatchRun(**self.conf)
            res=k.run()

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
