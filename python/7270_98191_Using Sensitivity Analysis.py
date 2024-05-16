from ipywidgets import widgets, interact
from IPython.display import display
get_ipython().magic('matplotlib inline')
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
figsize(12, 10)
sbn.set_context("talk", font_scale=1)

# The model used for this seminar is contained in the file model.py
from model import (cost_of_vehicle_to_grid, compute_profit, 
                   annualized_capital_cost, battery_lifetime, 
                   max_vehicle_power)

# Uncomment the next line and run this cell to view the model code in this notebook
# %load model.py

@interact(connector=widgets.FloatSlider(value=2.3, min=2.3, max=22, step=0.5), 
          battery_size=widgets.FloatSlider(value=24, min=10, max=100, step=5), 
          distance_driven=widgets.FloatSlider(value=0, min=0, max=100, step=5), 
          range_buffer=widgets.FloatSlider(value=0, min=0, max=100, step=10),
          dispatch_time=widgets.FloatSlider(value=1.4, min=0.5, max=24, step=0.5))
def plot_power(connector: float, battery_size: float, distance_driven: float, 
               range_buffer: float, dispatch_time: float) -> float :
    power = max_vehicle_power(connector,
                      battery_size,
                      distance_driven,
                      range_buffer,
                      dispatch_time
                      )
    return print("The maximum power is {} kW".format(round(power, 2)))

def monte_carlo_large(data):
    dispatch_time = 4
    y = max_vehicle_power(data[0], data[1], data[2], data[3], data[6], data[4], data[5])
    return y

number_sims = 1000

# Make some random data in the correct ranges
mc_connector = np.random.uniform(2.3, 22, number_sims)
mc_battery_size = np.random.uniform(50, 100, number_sims)
mc_distance_driven = np.random.uniform(0, 80, number_sims)
mc_range_buffer = np.random.uniform(0, 80, number_sims)
mc_driving_eff = np.random.uniform(2, 6, number_sims)
mc_inv_eff = np.random.uniform(0.87, 0.97, number_sims)
mc_dispatch_time = np.random.uniform(0.5, 24, number_sims)

data = np.array((mc_connector, 
                 mc_battery_size, 
                 mc_distance_driven, 
                 mc_range_buffer, 
                 mc_driving_eff, 
                 mc_inv_eff, 
                 mc_dispatch_time))

# Run the code
y = monte_carlo_large(data)

# Make some scatter plots to compare the results
plt.subplot(241)
plt.scatter(mc_connector, y)
plt.title("Connector size (kW)")
plt.ylabel("Max Power (kW)")
plt.subplot(242)
plt.scatter(mc_battery_size, y)
plt.title("Battery Size (kWh)")
# plt.ylabel("Max Power (kW)")
plt.subplot(243)
plt.scatter(mc_distance_driven, y)
plt.title("Distance Driven (km)")
# plt.ylabel("Max Power (kW)")
plt.subplot(244)
plt.scatter(mc_range_buffer, y)
plt.title("Range Buffer (km)")
# plt.ylabel("Max Power (kW)")
plt.subplot(245)
plt.scatter(mc_driving_eff, y)
plt.title("Driving Eff (kWh/km)")
plt.ylabel("Max Power (kW)")
plt.subplot(246)
plt.scatter(mc_inv_eff, y)
plt.title("Inverter Eff (%)")
# plt.ylabel("Max Power (kW)")
plt.subplot(247)
plt.scatter(mc_dispatch_time, y)
plt.title("Dispatch Time (hours)")
# plt.ylabel("Max Power (kW)")
plt.tight_layout()
# plt.savefig('scatter.png')

plt.hist(y)
plt.xlabel("Power (kW)")
plt.ylabel("Frequency")

from SALib.sample import morris as ms
from SALib.analyze import morris as ma
from SALib.plotting import morris as mp

morris_problem = {
    # There are six variables
    'num_vars': 7,
    # These are their names
    'names': ['conn', 'batt', 'dist', 'range', 
              'dri_eff', 'inv_eff', 'dispatch_time'],
    # Plausible ranges over which we'll move the variables
    'bounds': [[2.3, 22], # connection_power (kW)
               [50, 100], # battery size (kWh)
               [0, 80], # distance driven (km)
               [0, 80], # range buffer (km)
               [4,5.5], # driving efficiency (kWh/km)
               [0.87,0.97], # inverter efficienct (%)
               [0.5, 24] # dispatch time - hours of the day in which 
                         # the energy is dispatched
              ],
    # I don't want to group any of these variables together
    'groups': None
    }

number_of_trajectories = 1000
sample = ms.sample(morris_problem, number_of_trajectories, num_levels=4, grid_jump=2)

# Run the sample through the monte carlo procedure of the power model
output = monte_carlo_large(sample.T)
# Store the results for plotting of the analysis
Si = ma.analyze(morris_problem, sample, output, print_to_console=False)
print("{:20s} {:>7s} {:>7s} {:>7s}".format("Name", "mu", "mu_star", "sigma"))
for name, s1, st, mean in zip(morris_problem['names'], 
                              Si['mu'], 
                              Si['mu_star'], 
                              Si['sigma']):
    print("{:20s} {:=7.2f} {:=7.2f} {:=7.2f}".format(name, s1, st, mean))

fig, (ax1, ax2) = plt.subplots(1,2)
mp.horizontal_bar_plot(ax1, Si, {})
mp.covariance_plot(ax2, Si, {})

@interact(battery_size=widgets.FloatSlider(value=24, min=10, max=100, step=5), 
          battery_unit_cost=widgets.FloatSlider(value=350, min=100, max=400, step=50),
          connector_power=widgets.FloatSlider(value=2.3, min=2.3, max=22, step=0.5), 
          lifetime_cycles=widgets.FloatSlider(value=2000, min=1000, max=10000, step=1000),
          depth_of_discharge=widgets.FloatSlider(value=0.8, min=0.5, max=1.0, step=0.1),
          electricity_price=widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01),
          purchased_energy_cost=widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01),
          capacity_price=widgets.FloatSlider(value=0.007, min=0.001, max=0.01, step=0.001),
          round_trip_efficiency=widgets.FloatSlider(value=0.73, min=0.50, max=1.0, step=0.01),
          cost_of_v2g_equip=widgets.FloatSlider(value=2000, min=100, max=5000, step=100),
          discount_rate=widgets.FloatSlider(value=0.10, min=0.0, max=0.2, step=0.01),
          economic_lifetime=widgets.FloatSlider(value=10, min=3, max=25, step=1),
          ratio_dispatch_to_contract=widgets.FloatSlider(value=0.10, min=0.01, max=0.50, step=0.01),
          distance_driven=widgets.FloatSlider(value=0, min=0, max=100, step=5), 
          range_buffer=widgets.FloatSlider(value=0, min=0, max=100, step=10),
          hours_connected_per_day=widgets.FloatSlider(value=18, min=0.5, max=24, step=0.5))
def plot_profit(battery_size,
                battery_unit_cost,
                connector_power,
                lifetime_cycles,
                depth_of_discharge,
                electricity_price,
                purchased_energy_cost,
                capacity_price,
                round_trip_efficiency,
                cost_of_v2g_equip,
                discount_rate,
                economic_lifetime,
                distance_driven,
                range_buffer,
                ratio_dispatch_to_contract,
                hours_connected_per_day):
    profit, revenue, cost = compute_profit(battery_size,
                                           battery_unit_cost,
                                           connector_power,
                                           lifetime_cycles,
                                           depth_of_discharge,
                                           electricity_price,
                                           purchased_energy_cost,
                                           capacity_price,
                                           round_trip_efficiency,
                                           cost_of_v2g_equip,
                                           discount_rate,
                                           economic_lifetime,
                                           distance_driven,
                                           range_buffer,
                                           ratio_dispatch_to_contract,
                                           hours_connected_per_day
                                           )
    return print("Profit £{} = £{} - £{}".format(np.round(profit,2), np.round(revenue, 2), np.round(cost,2) ))

from SALib.sample.saltelli import sample as ss
from SALib.analyze.sobol import analyze as sa

problem = {
    # There are sixteen variables
    'num_vars': 16,
    # These are their names
    'names': ['battery_size',
              'battery_unit_cost',
              'connector_power',
              'lifetime_cycles',
              'depth_of_discharge',
              'electricity_price',
              'purchased_energy_cost',
              'capacity_price',
              'round_trip_efficiency',
              'cost_of_v2g_equip',
              'discount_rate',
              'economic_lifetime',
              'distance_driven',
              'range_buffer',
              'ratio_dispatch_to_contract',
              'hours_connected_per_day'],
    # These are their plausible ranges over which we'll move the variables
    'bounds': [       
                [10, 100],
                [100, 400],
                [2.3, 22],
                [1000, 10000],
                [0.5, 1.0],
                [0.01, 0.2], 
                [0.01, 0.2],
                [0.001, 0.01], 
                [0.65, 1.0],
                [100, 5000],
                [0.0, 0.2], 
                [3, 25],
                [0, 100], 
                [0, 100], 
                [0.01, 0.50],
                [0.5, 24],
              ],
    # I don't want to group any of these variables together
    'groups': None
    }

sample = ss(problem, 1000, calc_second_order=False)
profit, revenue, cost = compute_profit(sample[:, 0], sample[:, 1], sample[:, 2], 
                                       sample[:, 3], sample[:, 4], sample[:, 5], 
                                       sample[:, 6], sample[:, 7], sample[:, 8], 
                                       sample[:, 9], sample[:, 10], sample[:, 11], 
                                       sample[:, 12], sample[:, 13], sample[:, 14], 
                                       sample[:, 15])
SI = sa(problem, profit, calc_second_order=False, print_to_console=False)
print("{:28s} {:>5s} {:>5s} {:>12s}".format("Name", "1st", "Total", "Mean of Input"))
for name, s1, st, mean in zip(problem['names'], SI['S1'], SI['ST'], sample.mean(axis=0)):
    print("{:28s} {:=5.2f} {:=5.2f} ({:=12.2f})".format(name, s1, st, mean))

@interact(battery_size=widgets.FloatSlider(value=70, min=10, max=100, step=5), 
          purchased_energy_cost=widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01),
          cost_of_v2g_equip=widgets.FloatSlider(value=2000, min=100, max=5000, step=100),
          economic_lifetime=widgets.FloatSlider(value=10, min=3, max=25, step=1),
          ratio_dispatch_to_contract=widgets.FloatSlider(value=0.10, min=0.01, max=0.50, step=0.01),
         lifetime_cycles=widgets.FloatSlider(value=2000, min=1000, max=10000, step=500))
def plot_profit(battery_size,
                purchased_energy_cost,
                cost_of_v2g_equip,
                economic_lifetime,
                ratio_dispatch_to_contract,
                lifetime_cycles):
    profit, revenue, cost = compute_profit(lifetime_cycles=lifetime_cycles,
                                           battery_size=battery_size,
                                           purchased_energy_cost=purchased_energy_cost,
                                           cost_of_v2g_equip=cost_of_v2g_equip,
                                           economic_lifetime=economic_lifetime,
                                           ratio_dispatch_to_contract=ratio_dispatch_to_contract
                                           )
    return print("Profit £{} = £{} - £{}".format(np.round(profit,2), 
                                                 np.round(revenue, 2), 
                                                 np.round(cost,2) ))

