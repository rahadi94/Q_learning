import numpy as np
import pandas as pd

class RL_agent:
    def __init__(self, env, episode):
        self.env = env
        SOC = range(11)
        time = range(24)
        position = range(89)
        supply = range(4)
        queue = range(2)
        waiting_list = range(4)
        if episode == 0:
            index = pd.MultiIndex.from_product([SOC, time, position, supply, queue, waiting_list],
                                               names=['SOC', 'time', 'position', 'supply', 'queue', 'waiting_list'])
            self.q_table = pd.DataFrame(-np.random.rand(len(index), 3), index=index)
            self.q_table['counter'] = 0
        else:
            self.q_table = pd.read_csv('q_table.csv')
            self.q_table = self.q_table.set_index(['SOC', 'time', 'position', 'supply', 'queue', 'waiting_list'])
        '''if episode == 1:
            self.q_table['counter'] = 0'''

    def get_state(self, vehicle, charging_station, vehicles, waiting_list):
        SOC = int((vehicle.charge_state - vehicle.charge_state % 10) / 10)
        if isinstance(SOC, np.ndarray):
            SOC = SOC[0]
        for j in range(0, 24):
            if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                hour = j
        position = vehicle.position.id
        supply = len([v for v in vehicles if v.location.distance_1(vehicle.location) <= 4 and v.charge_state >= 30 and
                      vehicle.mode in ['idle', 'parking', 'circling', 'queue']])
        if supply == 0:
            supply = 0
        elif supply < 5:
            supply = 1
        elif supply < 10:
            supply = 2
        else:
            supply = 3
        if isinstance(supply, np.ndarray):
            supply = supply[0]
        wl = len([t for t in waiting_list if t.origin.distance_1(vehicle.location) <= 5])
        if wl == 0:
            wl = 0
        elif wl < 5:
            wl = 1
        elif wl < 10:
            wl = 2
        else:
            wl = 3
        if isinstance(wl, np.ndarray):
            wl = wl[0]
        q = len(charging_station.plugs.queue)
        if q == 0:
            queue = 0
        else:
            queue = 1
        if isinstance(queue, np.ndarray):
            queue = queue[0]
        return (SOC, hour, position, supply, queue, wl)

    def take_action(self, vehicle, charging_station, vehicles, waiting_list):
        epsilon = 0.1
        state = self.get_state(vehicle, charging_station, vehicles, waiting_list)
        if np.random.random() > epsilon:
            if vehicle.charge_state >= 70:
                action = np.argmax(self.q_table.loc[state, ['0', '1', '2']])
            else:
                action = np.argmax(self.q_table.loc[state, ['0', '1']])
        else:
            if vehicle.charge_state >= 70:
                action = np.random.randint(0, 3)
            else:
                action = np.random.randint(0, 2)
        vehicle.old_state = state
        # print(f'state is {vehicle.old_state} and action is {action}')
        vehicle.old_action = action
        vehicle.old_time = self.env.now
        vehicle.reward['revenue'] = 0

        return action

    def update_value(self, vehicle, charging_station, vehicles, waiting_list):
        alpha = 0.1
        GAMMA = 0.5
        state = self.get_state(vehicle, charging_station, vehicles, waiting_list)
        if vehicle.charge_state >= 70:
            q = float(max(self.q_table.loc[state, ['0', '1', '2']]))
        else:
            q = float(max(self.q_table.loc[state, ['0', '1']]))

        vehicle.r = -(vehicle.reward['charging'] + vehicle.reward['distance'] * 0.80 - vehicle.reward[
            'revenue'] + vehicle.reward['queue'] / 30)
        vehicle.total_rewards['state'].append(vehicle.old_state)
        vehicle.total_rewards['action'].append(vehicle.old_action)
        vehicle.total_rewards['reward'].append(vehicle.r)
        vehicle.profit += vehicle.r

        vehicle.old_q = self.q_table.loc[vehicle.old_state, f'{vehicle.old_action}']

        '''print(f'state is {state}')
        print(f'old_q is {vehicle.old_q}')
        print(f'q is {q}')
        print(f'r is {vehicle.reward}')
        print(f'q{vehicle.old_state} is updated')'''
        self.q_table.loc[vehicle.old_state, f'{vehicle.old_action}'] = vehicle.old_q + \
                                                                      alpha * (vehicle.r + GAMMA * q - vehicle.old_q)
        self.q_table.loc[vehicle.old_state, 'counter'] += 1

