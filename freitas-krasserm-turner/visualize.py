import matplotlib.pyplot as plt
import numpy as np

data_file = 'baryons-from-mesons/GaussianProcess/data/mesoninputs.dat'
baryon_file = 'baryons-from-mesons/GaussianProcess/data/baryoninputs.dat'
target_file = 'baryons-from-mesons/GaussianProcess/data/mMass.dat'
baryon_target_file = 'baryons-from-mesons/GaussianProcess/data/bMass.dat'
light_mesons_file = 'lightmesons.dat'
heavy_mesons_file = 'heavymesons.dat'
light_baryons_file = 'lightbaryons.dat'
heavy_baryons_file = 'heavybaryons.dat'

target = np.loadtxt(target_file)
data = np.loadtxt(data_file)
baryon_data = np.loadtxt(baryon_file)
baryon_targets = np.loadtxt(baryon_target_file)
light_mesons = np.loadtxt(light_mesons_file)
light_baryons = np.loadtxt(light_baryons_file)
heavy_mesons = np.loadtxt(heavy_mesons_file)
heavy_baryons = np.loadtxt(heavy_baryons_file)

# Identify index from main set to find corresponding masses of light/heavy mesons
def find_index(query_arr, source_arr):
    return np.where((source_arr == query_arr[:, None]).all(-1))[1]

# Light Mesons
light_meson_idx = find_index(light_mesons, data)
light_meson_masses = np.exp(target[light_meson_idx])
# Heavy Mesons
heavy_meson_idx = find_index(heavy_mesons, data)
heavy_meson_masses = np.exp(target[heavy_meson_idx])
# Light Baryons
light_baryon_idx = find_index(light_baryons, baryon_data)
light_baryon_masses = np.exp(baryon_targets[light_baryon_idx])
# Heavy Baryons
heavy_baryon_idx = find_index(heavy_baryons, baryon_data)
heavy_baryon_masses = np.exp(baryon_targets[heavy_baryon_idx])

# Use last meson mass as dividing line
font = {'size': 16}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)

plt.scatter(light_meson_masses, np.zeros_like(light_meson_masses), marker='.', c='tab:orange', label='Light Meson')
plt.scatter(heavy_meson_masses, np.zeros_like(heavy_meson_masses), marker='x', c='tab:blue', label='Heavy Meson')
plt.scatter(light_baryon_masses, np.zeros_like(light_baryon_masses), marker='x', c='tab:green', label='Light Baryon')
plt.scatter(heavy_baryon_masses, np.zeros_like(heavy_baryon_masses), marker='.', c='tab:pink', label='Heavy Baryon')
plt.axvline(np.max(light_meson_masses))
# plt.annotate('P1', (np.max(light_meson_masses), 0))
plt.legend()
plt.show()
