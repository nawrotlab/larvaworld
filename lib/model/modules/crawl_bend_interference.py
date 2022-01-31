import numpy as np


class Oscillator_coupling():
    def __init__(self,locomotor, crawler_phi_range=[0.0, 0.0],feeder_phi_range=[0.0, 0.0],attenuation=0.0):
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range
        self.attenuation = attenuation
        self.locomotor = locomotor
        self.crawler = locomotor.crawler
        self.feeder = locomotor.feeder

    def step(self):
        if self.crawler is not None:
            if self.crawler.effector:
                phi = self.crawler.phi / np.pi
                p0, p1 = self.crawler_phi_range
                if self.crawler.waveform in ['realistic', 'gaussian'] and (phi < p0 or phi > p1):
                    return True
                elif self.crawler.waveform == 'square' and not phi <= 2 * self.crawler.square_signal_duty:
                    return True
                elif self.crawler.waveform == 'constant':
                    return True


        if self.feeder is not None:
            if self.feeder.effector:
                phi = self.feeder.phi / np.pi
                p0, p1 = self.feeder_phi_range
                if p0 < phi < p1:
                    return True
        return False