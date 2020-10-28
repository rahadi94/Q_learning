import numpy as np


class RL_agent:
    def __init__(self, env):
        self.env = env
        self.q_table = dict()
        for i in range(10):
            for ii in range(24):
                for iii in range(89):
                    for iiii in range(50):
                        for iiiii in range(50):
                            state = dict(SOC=i, time=ii, position=iii, supply=iiii, waiting_list=iiiii)
                            self.q_table[state['SOC'], state['time'], state['position'],
                                         state['supply'], state['waiting_list']] = [round(np.random.uniform(-1, 0), 2)
                                                                                    for i in range(2)]

    def get_state(self, vehicle, vehicles, waiting_list):
        for i in range(10):
            if i * 10 <= vehicle.charge_state <= (i + 1) * 10:
                SOC = i
        for j in range(0, 24):
            if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                hour = j
        position = vehicle.position.id
        supply = len([v for v in vehicles if v.location.distance_1(vehicle.location) <= 2 and v.charge_state >= 30])
        wl = len([t for t in waiting_list if t.origin.distance_1(vehicle.location) <= 2])
        return (SOC, hour, position, supply, wl)

    def take_action(self, vehicle, vehicles, waiting_list):
        epsilon = 0.1
        state = self.get_state(vehicle, vehicles, waiting_list)
        if np.random.random() > epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(0, 1)
        vehicle.old_state = state
        vehicle.old_q = self.q_table[state][action]
        vehicle.old_action = action
        vehicle.old_time = self.env.now
        vehicle.reward['missed_trips'] = 0
        return action

    def update_value(self, vehicle, vehicles, waiting_list):
        alpha = 0.1
        GAMMA = 0.5
        state = self.get_state(vehicle, vehicles, waiting_list)
        q = max(self.q_table[state])
        vehicle.r = vehicle.reward['charging'] / 100 + vehicle.reward['distance'] / 10 + vehicle.reward[
            'missed_trips']
        self.q_table[vehicle.old_state][vehicle.old_action] = vehicle.old_q + \
                                                              alpha * (vehicle.r + GAMMA * q - vehicle.old_q)
