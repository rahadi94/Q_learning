from Fleet_sim.location import find_zone
import numpy as np
from Fleet_sim.read import zones
import logging

lg = logging.getLogger(__name__)
lg.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s:%(message)s')

file_handler = logging.FileHandler('report.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

lg.addHandler(file_handler)
lg.addHandler(stream_handler)


class Vehicle:
    speed = 0.5  # km/min
    charging_cost = [32.10, 32.00, 30.58, 30.02, 30.27, 33.19, 37.32, 44.45, 46.82, 45.70, 40.60,
                     35.59, 34.08, 33.85, 34.45, 39.58, 45.84, 49.26, 52.40, 49.07, 43.55, 38.25, 38.10, 34.44]
    parking_cost = 5

    # remove everything env related and put it into VehicleSimulation

    def __init__(self, id, env, initial_location, capacity, charge_state, mode):
        self.env = env
        self.info = dict()
        self.info['SOC'] = []
        self.info['location'] = []
        self.info['position'] = []
        self.info['mode'] = []
        self.location = initial_location
        self.id = id
        self.mode = mode
        self.position = find_zone(self.location, zones)
        """Allowed modes are:
             active - car is currently driving a passenger from pickup to destination
             locked - car is currently going to pickup location to pick up customer
             idle - car is currently idle and waiting for request
             relocating - car is moving to a different comb
             charging - car is currently charging
             en_route_to_charge - car is on its way to a charging station"""
        self.battery_capacity = capacity
        self.charge_state = charge_state
        self.count_request_accepted = 0
        self.rental_time = 0.0
        self.fuel_consumption = 0.20  # in kWh/km
        '''self.count_times = dict()
        self.count_times['active'] = 0
        self.count_times['locked'] = 0
        self.count_times['idle'] = 1
        self.count_times['relocating'] = 0
        self.count_times['charging'] = 0
        self.count_times['parking'] = 0
        self.count_times['ertc'] = 0
        self.count_times['ertp'] = 0
        self.count_seconds = dict()
        self.count_seconds['active'] = 0.0
        self.count_seconds['locked'] = 0.0
        self.count_seconds['idle'] = 0.0
        self.count_seconds['relocating'] = 0.0
        self.count_seconds['charging'] = 0.0
        self.count_seconds['parking'] = 0.0
        self.count_seconds['ertc'] = 0.0
        self.count_seconds['ertp'] = 0.0
        self.last_count_seconds_idle = 0.0
        self.last_count_seconds_relocating = 0.0
        self.count_km = dict()
        self.count_km['active'] = 0.0
        self.count_km['locked'] = 0.0
        self.count_km['relocating'] = 0.0
        self.count_km['ertc'] = 0.0
        self.count_km['ertp'] = 0.0'''
        self.t_start_charging = None
        self.t_arriving_CS = None
        self.costs = dict()
        self.costs['charging'] = 0.0
        self.costs['parking'] = 0.0
        self.charge_consumption_dropoff = None
        self.time_to_pickup = None
        self.charge_consumption_pickup = None
        self.distance_to_CS = None
        self.time_to_CS = None
        self.charging_threshold = None
        self.charge_duration = None
        self.time_to_relocate = None
        self.charge_consumption_relocate = None
        self.distance_to_parking = None
        self.time_to_parking = None
        self.parking_stop = env.event()
        self.circling_stop = env.event()
        self.charging_interruption = env.event()
        self.charging_start = env.event()
        self.charging_end = env.event()
        self.charging_interrupt = env.event()
        self.trip_end = env.event()
        self.relocating_end = env.event()
        self.queue_interruption = env.event()
        self.reward = dict(charging=0, queue=0, distance=0, revenue=0)
        self.old_action = 0
        self.old_state = (0, 0, 0, 0, 0, 0)
        self.charging_count = 0

    def SOC_consumption(self, distance):
        return float(distance * self.fuel_consumption * 100.0 / self.battery_capacity)

    def send(self, trip):
        self.mode = 'locked'
        distance_duration = self.location.distance(trip.origin)
        distance_to_pickup = distance_duration[0]
        distance_to_dropoff = trip.distance
        self.time_to_pickup = distance_duration[1]
        self.charge_consumption_pickup = self.SOC_consumption(distance_to_pickup)
        self.charge_consumption_dropoff = self.SOC_consumption(distance_to_dropoff)

        self.rental_time = trip.duration
        lg.info(f'Vehicle {self.id} is sent to the request {trip.id}')
        # print(f'Vehicle {self.id} is sent to the request {trip.id}')

        '''self.task_list.append({'mode': 'locked',
                               'duration': self.time_to_pickup,
                               'start time': trip.start_time,
                               'end time': self.env.now + self.time_to_pickup})
        self.task_list.append({'mode': 'active',
                               'duration': self.rental_time,
                               'start time': self.time_to_pickup,
                               'end time': trip.start_time + self.time_to_pickup + self.rental_time})

        self.count_seconds['idle'] += trip.interarrival
        self.count_seconds['locked'] += self.time_to_pickup
        self.count_seconds['active'] += self.rental_time
        self.count_times['locked'] += 1
        self.count_times['active'] += 1
        self.count_km['locked'] += distance_to_pickup
        self.count_km['active'] += distance_to_dropoff
        self.count_request_accepted += 1'''

    def pick_up(self, trip):
        self.mode = 'active'
        lg.info(f'Vehicle {self.id} pick up the user {trip.id} at {self.env.now}')
        # print(f'Vehicle {self.id} pick up the user {trip.id} at {self.env.now}')
        self.charge_state -= self.charge_consumption_pickup
        trip.info['pickup_time'] = self.env.now
        trip.info['waiting_time'] = trip.info['pickup_time'] - trip.info['arrival_time']
        self.location = trip.origin

    def drop_off(self, trip):
        self.mode = 'idle'
        self.charge_state -= self.charge_consumption_dropoff
        self.location = trip.destination
        self.position = find_zone(self.location, zones)
        # self.count_times['idle'] += 1
        lg.info(f'Vehicle {self.id} drop off the user {trip.id} at {self.env.now}')
        # print(f'Vehicle {self.id} drop off the user {trip.id} at {self.env.now}')

    def send_charge(self, charging_station):
        self.mode = 'ertc'
        lg.info(f'Charging state of vehicle {self.id} is {self.charge_state}')
        lg.info(f'Vehicle {self.id} is sent to the charging station {charging_station.id} at {self.env.now}')
        # print(f'Charging state of vehicle {self.id} is {self.charge_state}')
        # print(f'Vehicle {self.id} is sent to the charging station {charging_station.id} at {self.env.now}')
        distance_duration = self.location.distance(charging_station.location)
        self.distance_to_CS = distance_duration[0]
        self.reward['distance'] = distance_duration[0]
        self.time_to_CS = distance_duration[1]


        self.position = find_zone(self.location, zones)
        """self.task_list.append({'mode': 'ertc',
                               'duration': self.time_to_CS,
                               'start time': self.env.now,
                               'end time': self.env.now + self.time_to_CS})
        self.count_times['ertc'] += 1"""

    def charging(self, charging_station):
        self.mode = 'charging'
        charge_consumption_to_charging = self.SOC_consumption(self.distance_to_CS)
        self.charge_state -= charge_consumption_to_charging
        time = self.env.now
        if time % 1440 < 0.25 * 1440:
            self.charging_threshold = 100
        elif time % 1440 < 0.50 * 1440:
            self.charging_threshold = 100
        elif time % 1440 < 0.75 * 1440:
            self.charging_threshold = 100
        else:
            self.charging_threshold = 100
        self.charge_duration = (((self.charging_threshold - self.charge_state) * self.battery_capacity / 100)
                                / charging_station.power)
        self.location = charging_station.location
        self.position = find_zone(self.location, zones)
        lg.info(f'Vehicle {self.id} enters the station at {self.env.now}')
        # print(f'Vehicle {self.id} enters the station at {self.env.now}')
        """self.task_list.append({'mode': 'charging',
                                       'duration': self.charge_duration,
                                       'start time': self.env.now,
                                       'end time': self.env.now + self.charge_duration})
        self.count_times['charging'] += 1
        self.count_seconds['ertc'] += self.time_to_CS
        self.count_km['ertc'] += self.distance_to_CS"""

    def finish_charging(self, charging_station):
        self.mode = 'idle'
        for j in range(0, 24):
            if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                h = j
        self.reward['charging'] = (self.charging_threshold - self.charge_state) / 100 * 50 * self.charging_cost[h]/100
        if isinstance(self.reward['charging'], np.ndarray):
            self.reward['charging'] = self.reward['charging'][0]
        self.costs['charging'] += (self.charging_threshold - self.charge_state) / 100 * 50 * self.charging_cost[h]/100
        self.charge_state += (charging_station.power * self.charge_duration) / (self.battery_capacity / 100)
        '''if self.charge_state > 100:
            self.charge_state = 100'''
        lg.info(f'Finished charging, Charging state of vehicle {self.id} is {self.charge_state} at {self.env.now}')
        # print(f'Finished charging, Charging state of vehicle {self.id} is {self.charge_state} at {self.env.now} ')
        # self.count_seconds['charging'] += self.charge_duration

    def relocate(self, target_zone):
        distance_duration = self.location.distance(target_zone.centre)
        distance_to_target = distance_duration[0]

        # distance divided by speed to calculate pick up time

        self.time_to_relocate = distance_duration[1]
        self.charge_consumption_relocate = self.SOC_consumption(distance_to_target)
        lg.info(f'Vehicle {self.id} is relocated to the zone {target_zone.id}')
        # print(f'Vehicle {self.id} is relocated to the zone {target_zone.id}')
        self.mode = 'relocating'

        """self.task_list.append({'mode': 'relocating',
                               'duration': self.time_to_relocate,
                               'start time': self.env.now,
                               'end time': self.env.now + self.time_to_relocate})

        self.count_seconds['relocating'] += self.time_to_relocate
        self.count_times['relocating'] += 1
        self.count_km['relocating'] += distance_to_target"""

    def finish_relocating(self, target_zone):
        self.charge_state -= self.charge_consumption_relocate
        self.location = target_zone.centre
        self.position = find_zone(self.location, zones)
        self.mode = 'idle'

    def send_parking(self, parking):
        self.mode = 'ertp'
        if self.env.now != 0:
            lg.info(f'Vehicle {self.id} is sent to the parking {parking.id} at {self.env.now}')
            # print(f'Vehicle {self.id} is sent to the parking {parking.id} at {self.env.now}')
        distance_duration = self.location.distance(parking.location)
        self.distance_to_parking = distance_duration[0]
        self.time_to_parking = distance_duration[1]
        charge_consumption_to_parking = self.SOC_consumption(self.distance_to_parking)
        self.charge_state -= charge_consumption_to_parking
        '''self.count_times['ertp'] += 1
        self.count_seconds['ertp'] += self.time_to_parking
        self.count_km['ertp'] += self.distance_to_parking'''

    def parking(self, parking):
        self.mode = 'parking'
        # self.count_times['parking'] += 1
        self.location = parking.location
        self.position = find_zone(self.location, zones)
        if self.env.now >= 5:
            lg.info(f'Vehicle {self.id} start parking at {self.env.now}')
            # print(f'Vehicle {self.id} start parking at {self.env.now}')
