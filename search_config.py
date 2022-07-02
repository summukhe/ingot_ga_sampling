import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ga.impl.ingot import Ingot
from ga.base.ga_search import GASearch
from ga.base.selector import RouletteSelector
from ga.impl.chromosome_impl import IngotState
from ga.impl.operation_impl import IngotStateUpdate
from ga.impl.sampler_impl import IngotStateSampler
from ga.impl.plot_ingot import IngotStatePlotter

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')
data_file = os.path.join(data_dir, 'Cropped_point_cloud.pcd')
assert os.path.isfile(data_file)

pc = o3d.io.read_point_cloud(data_file)
data = np.array(pc.points, dtype='float32')

ingot_spec = {'top_diameter': 100,
              'base_diameter': 250,
              'cone_height': 0,
              'cylinder_height': 100,
              'allowed_error': 20}

x_min = data.min(axis=0)
x_max = data.max(axis=0)
scale = np.max(x_max - x_min) / ingot_spec['base_diameter']


gen = IngotStateSampler(scale_range=(0.25 * scale, 1.5 * scale),
                        min_base_plane=tuple(x_min),
                        max_base_plane=tuple(x_max),
                        min_axis_angle=(-np.pi/2, -np.pi/2),
                        max_axis_angle=(np.pi/2, np.pi/2))

updater = IngotStateUpdate(scale_range=(0.25 * scale, 1.5 * scale),
                           min_base_plane=tuple(x_min),
                           max_base_plane=tuple(x_max),
                           min_axis_angle=(-np.pi/2, -np.pi/2),
                           max_axis_angle=(np.pi/2, np.pi/2))

ingot_plotter = IngotStatePlotter(point_cloud=data)
plot_counter = 0
plot_interval = 10
with_plot = True


def visitor(c_state: IngotState):
    global plot_counter
    plot_counter += 1
    if plot_counter % plot_interval == 0:
        fig = plt.figure()
        ingot = Ingot(scale=c_state.scale,
                      z_axis=c_state.z_axis,
                      base_center=c_state.base,
                      **ingot_spec)
        ingot_plotter.set_ingot_state(ingot)
        ingot_plotter.plot(fig)
        plt.show()


def fun(c_state: IngotState):
    ingot = Ingot(scale=c_state.scale,
                  z_axis=c_state.z_axis,
                  base_center=c_state.base,
                  **ingot_spec)
    return ingot.likelihood(data)


search = GASearch(fitness=fun,
                  n_population=100,
                  n_iteration=50,
                  generator=gen,
                  selector=RouletteSelector(max_partitions=10),
                  operator=updater,
                  n_elites=5,
                  visitor=visitor
                  )

res = search.search(verbose=True)

print(res)
