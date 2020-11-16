import pandas as pd

from Fleet_sim.Zone import Zone


demand_table = pd.read_csv('demand_table.csv')
OD_table = pd.read_csv('origin_destination.csv')

z = 0
zones = list()
for hex in demand_table['h3_hexagon_id_start'].values:
    z += 1
    '''demand = (demand_table[demand_table['h3_hexagon_id_start'] == hex]
                  .drop('h3_hexagon_id_start', axis=1))/1440 + 0.001'''
    demand = (demand_table[demand_table['h3_hexagon_id_start'] == hex]).drop('h3_hexagon_id_start', axis=1)
    destination = (OD_table[OD_table['h3_hexagon_id_start'] == hex]
                   .drop('h3_hexagon_id_start', axis=1)).sort_values(by=z - 1, axis=1).T.reset_index()
    zone = Zone(z, hex, demand, destination)
    zones.append(zone)
charging_threshold = [40, 45, 50, 55, 52, 50, 48, 45, 45, 42, 40, 40, 40, 40, 40, 38, 35, 32, 30, 30, 27, 30, 32, 35]

charging_cost = [32.10, 32.00, 30.58, 30.02, 30.27, 33.19, 37.32, 44.45, 46.82, 45.70, 40.60,
                     35.59, 34.08, 33.85, 34.45, 39.58, 45.84, 49.26, 52.40, 49.07, 43.55, 38.25, 38.10, 34.44]
