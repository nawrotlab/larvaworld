import agentpy
import numpy as np
from shapely import geometry

from larvaworld.lib import reg, aux


class Arena(agentpy.Space):
    def __init__(self, model, dims, shape= 'rectangular',vertices=None, torus=False):
        X, Y = self.dims = np.array(dims)
        if vertices is None:
            if shape == 'circular':
                # This is a circle_to_polygon shape from the function
                vertices = aux.circle_to_polygon(60, X / 2)
            elif shape == 'rectangular':
                # This is a rectangular shape
                vertices = np.array([(-X / 2, -Y / 2),
                                   (-X / 2, Y / 2),
                                   (X / 2, Y / 2),
                                   (X / 2, -Y / 2)])
            else:
                raise
        self.vertices =vertices
        self.range = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
        k = 0.96
        self.polygon = geometry.Polygon(self.vertices * k)
        self.edges = [[geometry.Point(x1,y1), geometry.Point(x2,y2)] for (x1,y1), (x2,y2) in aux.group_list_by_n(vertices, 2)]
        super().__init__(model=model, torus=torus, shape=dims)

        self.stable_source_positions=[]
        self.displacable_source_positions=[]
        self.displacable_sources=[]
        self.stable_sources=[]
        self.accessible_sources =None
        self.accessible_sources_sorted = None
    # @staticmethod
    # def _border_behavior(position, shape, torus):
    #     # Border behavior
    #
    #     # Connected - Jump to other side
    #     if torus:
    #         for i in range(len(position)):
    #             while position[i] > shape[i]/2:
    #                 position[i] -= shape[i]
    #             while position[i] < -shape[i]/2:
    #                 position[i] += shape[i]
    #
    #     # Not connected - Stop at border
    #     else:
    #         for i in range(len(position)):
    #             if position[i] > shape[i]/2:
    #                 position[i] = shape[i]/2
    #             elif position[i] < -shape[i]/2:
    #                 position[i] = -shape[i]/2

    def place_agent(self, agent, pos):
        pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
        self.positions[agent] = pos  # Add pos to agent_dict

    def move_agent(self, agent, pos):
        self.move_to(agent, pos)

    def add_sources(self, sources, positions):
        for source, pos in zip(sources, positions):
            pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
            if source.can_be_displaced :
                self.displacable_source_positions.append(pos)  # Add pos to agent_dict
                self.displacable_sources.append(source)  # Add pos to agent_dict
            else :
                self.stable_source_positions.append(pos)
                self.stable_sources.append(source)

    def source_positions_in_array(self):
        if len(self.displacable_sources)>0 :
            for i,source in enumerate(self.displacable_sources):
                self.displacable_source_positions[i]=np.array(source.get_position())
            if len(self.stable_sources)>0 :
                self.source_positions=np.vstack([np.array(self.displacable_source_positions),np.array(self.stable_source_positions)])
                self.sources=np.array(self.displacable_sources+self.stable_sources)
            else :
                self.source_positions = np.array(self.displacable_source_positions)
                self.sources = np.array(self.displacable_sources)
        else :
            self.source_positions = np.array(self.stable_source_positions)
            self.sources = np.array(self.stable_sources)

    def accesible_sources(self, pos, radius):
        return self.sources[np.where(aux.eudi5x(self.source_positions, pos) <= radius)].tolist()

    def accessible_sources_multi(self, agents, positive_amount=True, return_closest=True):
        # if self.accessible_sources_sorted is None and self.accessible_sources is None:
        self.source_positions_in_array()
        if positive_amount :
            idx=np.array([s.amount > 0 for s in self.sources])
            self.sources=self.sources[idx]
            self.source_positions=self.source_positions[idx]
        ps=np.array(agents.pos)
        ds=aux.eudiNxN(self.source_positions, ps)
        self.accessible_sources_sorted={a: {'sources' : self.sources[np.argsort(ds[i])], 'dsts': np.sort(ds[i])} for i,a in enumerate(agents)}
        if not return_closest:
            dic={a: dic['sources'][dic['dsts']<=a.radius].tolist() for a, dic in self.accessible_sources_sorted.items()}
            # dic={a: self.sources[np.where(ds[i] <= a.radius)].tolist() for i,a in enumerate(agents)}
        else:
            dic={a: dic['sources'][0] if dic['dsts'][0]<=a.radius else None for a, dic in self.accessible_sources_sorted.items()}
            # dic = {a: self.sources[np.argmin(ds[i])] if np.min(ds[i])<=a.radius else None for i, a in enumerate(agents)}
        self.accessible_sources = dic
        # else :
        #     if return_closest:
        #         for a, dic in self.accessible_sources_sorted.items() :
        #             for i in range(dic['sources'].shape[0]):
        #                 dic['dsts'][i]=aux.eudis5(a.pos, dic['sources'][i].pos)
        #                 if dic['dsts'][i] <= a.radius:
        #                     self.accessible_sources[a]=dic['sources'][i]
        #                     break
        #             self.accessible_sources[a]=None

