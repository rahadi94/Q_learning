import pandas as pd
import simpy
from Fleet_sim.location import find_zone
from Fleet_sim.trip import Trip
from Fleet_sim.Q_learner import RL_agent


# This function give us the available vehicles for a trip
def available_vehicle(vehicles, trip, SOC_threshold=20, max_distance=10):
    available_vehicles = list()
    for vehicle in vehicles:
        distance_to_pickup = vehicle.location.distance_1(trip.origin)
        distance_to_dropoff = trip.distance
        charge_consumption = (distance_to_pickup + distance_to_dropoff) * \
                             vehicle.fuel_consumption * 100.0 / vehicle.battery_capacity
        # Add idle vehicles that have enough energy to respond the trip into available vehicles and are not far away
        # too much
        if distance_to_pickup <= max_distance:
            if charge_consumption + SOC_threshold <= vehicle.charge_state and vehicle.mode in ['idle', 'parking',
                                                                                               'circling']:
                available_vehicles.append(vehicle)
    return available_vehicles


class Model:

    def __init__(self, env, vehicles, charging_stations, zones, parkings, simulation_time=500):
        self.t = []
        self.parkings = parkings
        self.zones = zones
        self.charging_stations = charging_stations
        self.vehicles = vehicles
        self.waiting_list = []
        self.simulation_time = simulation_time
        self.env = env
        self.trip_start = env.event()
        self.demand_generated = []
        self.utilization = []
        self.vehicle_id = None
        self.learner = RL_agent(env)

    def park(self, vehicle, parking):
        if self.env.now <= 5:
            vehicle.parking(parking)
            yield vehicle.parking_stop
        else:
            vehicle.mode = 'circling'
            print(f'vehicle {vehicle.id} start circling at {self.env.now}')
            circling_interruption = vehicle.circling_stop
            circling_finish = self.env.timeout(10)
            parking_events = yield circling_interruption | circling_finish
            if circling_interruption in parking_events:
                print(f'vehicle {vehicle.id} interrupt circling at {self.env.now}')
            if circling_finish in parking_events:
                print(f'vehicle {vehicle.id} stop circling at {self.env.now}')
                vehicle.send_parking(parking)
                yield self.env.timeout(vehicle.time_to_parking)
                vehicle.parking(parking)
                yield vehicle.parking_stop

    def parking_task(self, vehicle):
        if vehicle.mode in ['circling', 'idle']:
            # Finding the closest parking
            distances_to_PKs = [vehicle.location.distance_1(PK.location) for PK in self.parkings]
            parking = [x for x in self.parkings
                       if x.location.distance_1(vehicle.location) == min(distances_to_PKs)][0]
            with parking.capacity.request() as req:
                yield req
                yield self.env.process(self.park(vehicle, parking))

    def relocate(self, vehicle, target_zone):
        vehicle.relocate(target_zone)
        yield self.env.timeout(vehicle.time_to_relocate)
        vehicle.finish_relocating(target_zone)
        vehicle.relocating_end.succeed()
        vehicle.relocating_end = self.env.event()

    def relocate_task(self, vehicle):
        for zone in self.zones:
            zone.update(self.vehicles)
        vehicle.position = find_zone(vehicle.location, self.zones)
        time = self.env.now
        for i in range(0, 24):
            if i * 60 <= time % 1440 <= (i + 1) * 60:
                hour = i
        if vehicle.charge_state >= 50 and vehicle.mode in ['idle', 'parking'] and len(
                vehicle.position.list_of_vehicles) >= vehicle.position.demand.iloc[0, hour]:
            target_zones = [z for z in self.zones if len(z.list_of_vehicles) <= z.demand.iloc[0, hour]]

            if len(target_zones) > 1:
                distances_to_zones = [vehicle.location.distance_1(z.centre) for z in target_zones]
                target_zone = [z for z in target_zones
                               if z.centre.distance_1(vehicle.location) == min(distances_to_zones)][0]
                if vehicle.mode == 'parking':
                    vehicle.parking_stop.succeed()
                    vehicle.parking_stop = self.env.event()
                self.env.process(self.relocate(vehicle, target_zone))

    def charging_interruption(self):
        while True:
            for vehicle in self.vehicles:
                if vehicle.mode == 'charging':
                    vehicle.position = find_zone(vehicle.location, self.zones)
                    vehicle.position.update(self.vehicles)
                    soc = vehicle.charge_state + \
                          (11 / 60 * (self.env.now - vehicle.t_start_charging) * 100) / vehicle.battery_capacity
                    if soc > 50 and len([t for t in self.waiting_list
                                         if t.mode == 'unassigned' and t.zone == vehicle.position]) >= 1:
                        vehicle.charging_interruption.succeed()
                        vehicle.charging_interruption = self.env.event()
            yield self.env.timeout(15)

    def start_charge(self, charging_station, vehicle):
        vehicle.send_charge(charging_station)
        charging_demand = dict(vehicle_id=vehicle.id, time_send=self.env.now,
                               time_start=self.env.now + vehicle.time_to_CS,
                               SOC=vehicle.charge_state, lat=vehicle.location.lat, long=vehicle.location.long,
                               v_hex=vehicle.position.hexagon,
                               CS_location=[charging_station.location.lat, charging_station.location.long],
                               v_position=vehicle.position.id, CS_position=charging_station.id,
                               distance=vehicle.location.distance(charging_station.location)[0])
        self.demand_generated.append(charging_demand)
        yield self.env.timeout(vehicle.time_to_CS)
        vehicle.t_start_charging = self.env.now
        vehicle.charging(charging_station)
        vehicle.charging_start.succeed()
        vehicle.charging_start = self.env.event()

    def finish_charge(self, charging_station, vehicle):
        try:
            yield self.env.timeout(vehicle.charge_duration)
            vehicle.finish_charging(charging_station)
            vehicle.charging_end.succeed()
            vehicle.charging_end = self.env.event()
        except simpy.Interrupt:
            vehicle.charge_state += (charging_station.power * (
                    self.env.now - vehicle.t_start_charging) * 100) / vehicle.battery_capacity
            vehicle.mode = 'idle'
            # vehicle.count_seconds['charging'] += self.env.now - vehicle.t_start_charging
            vehicle.charging_interrupt.succeed()
            vehicle.charging_interrupt = self.env.event()
            vehicle.costs['charging'] += (vehicle.charging_threshold - vehicle.charge_state) * vehicle.charging_cost
            print(f'Warning!!!Charging state of vehicle {vehicle.id} is {vehicle.charge_state} at {self.env.now} ')

    # Checking charge status for vehicles and send them to charge if necessary
    '''def charge_task(self, vehicle):
        time = self.env.now
        th_list = charging_threshold
        for i in range(0, 24):
            if i * 60 <= time % 1440 <= (i + 1) * 60:
                threshold = th_list[i]
        if vehicle.charge_state <= threshold and vehicle.mode == 'idle':
            # Finding the closest charging station

            distances_to_CSs = [vehicle.location.distance_1(CS.location) for CS in self.charging_stations]
            charging_station = [x for x in self.charging_stations
                                if x.location.distance_1(vehicle.location) == min(distances_to_CSs)][0]
            with charging_station.plugs.request() as req:
                yield self.env.process(self.start_charge(charging_station, vehicle))
                yield req
                charging = self.env.process(self.finish_charge(charging_station, vehicle))
                yield charging | vehicle.charging_interruption
                if not charging.triggered:
                    charging.interrupt()
                    print(f'Vehicle {vehicle.id} stop charging at {self.env.now}')'''

    def charge_task(self, vehicle):
        if self.env.now > 30:
            action = self.learner.take_action(vehicle, self.vehicles, self.waiting_list)
            if action == 0 and vehicle.mode == 'idle':
                # Finding the closest charging station
                distances_to_CSs = [vehicle.location.distance_1(CS.location) for CS in self.charging_stations]
                charging_station = [x for x in self.charging_stations
                                    if x.location.distance_1(vehicle.location) == min(distances_to_CSs)][0]
                with charging_station.plugs.request() as req:
                    yield self.env.process(self.start_charge(charging_station, vehicle))
                    yield req
                    charging = self.env.process(self.finish_charge(charging_station, vehicle))
                    yield charging | vehicle.charging_interruption
                    if not charging.triggered:
                        charging.interrupt()
                        print(f'Vehicle {vehicle.id} stop charging at {self.env.now}')
            self.learner.update_value(vehicle, self.vehicles, self.waiting_list)

    def take_trip(self, trip, vehicle):
        vehicle.send(trip)
        trip.mode = 'assigned'
        self.waiting_list.remove(trip)
        yield self.env.timeout(vehicle.time_to_pickup)
        vehicle.pick_up(trip)
        trip.mode = 'in vehicle'
        yield self.env.timeout(trip.duration)
        vehicle.drop_off(trip)
        vehicle.trip_end.succeed()
        vehicle.trip_end = self.env.event()
        self.vehicle_id = vehicle.id
        trip.mode = 'finished'

    def trip_task(self, trip):
        available_vehicles = available_vehicle(self.vehicles, trip)
        distances = [vehicle.location.distance_1(trip.origin) for vehicle in available_vehicles]
        # If there is no available vehicle, add the trip to the waiting list
        if len(available_vehicles) == 0:
            return
        # Assigning the closest available vehicle to the trip
        print(f'There is/are {len(available_vehicles)} available vehicle(s) for trip {trip.id}')
        vehicle = [x for x in available_vehicles
                   if x.location.distance_1(trip.origin) == min(distances)][0]
        if vehicle.mode == 'parking':
            vehicle.parking_stop.succeed()
            vehicle.parking_stop = self.env.event()
        if vehicle.mode == 'circling':
            vehicle.circling_stop.succeed()
            vehicle.circling_stop = self.env.event()
        self.env.process(self.take_trip(trip, vehicle))

    def trip_generation(self, zone):
        j = 0
        # Trips are being generated randomly and cannot be rejected
        while True:
            j += 1
            trip = Trip(self.env, [zone.id, j], zone)
            yield self.env.timeout(trip.interarrival)
            self.trip_start.succeed()
            self.trip_start = self.env.event()
            trip.info['arrival_time'] = self.env.now
            self.waiting_list.append(trip)
            print(f'Trip {trip.id} is received at {self.env.now}')
            trip.start_time = self.env.now

    def missed_trip(self):
        while True:
            for trip in self.waiting_list:
                if trip.mode == 'unassigned' and self.env.now > (trip.start_time + 15):
                    trip.mode = 'missed'
                    self.waiting_list.remove(trip)
                    for vehicle in self.vehicles:
                        if vehicle.location.distance_1(trip.origin) <= 15:
                            vehicle.reward['missed_trips'] += 1
                    print(f'trip {trip.id} is missed at {self.env.now}')
            yield self.env.timeout(1)

    def hourly_charging(self):
        while True:
            for vehicle in self.vehicles:
                if vehicle.mode in ['idle', 'parking']:
                    self.env.process(self.charge_task(vehicle))
            yield self.env.timeout(60)

    def run(self):
        while True:
            yield self.trip_start
            for trip in self.waiting_list:
                if trip.mode == 'unassigned':
                    self.trip_task(trip)
                    yield self.env.timeout(0.001)

    def run_vehicle(self, vehicle):
        while True:
            # All vehicles start from parking
            if self.env.now == 0:
                self.env.process(self.parking_task(vehicle))

            # event_trip_start = self.trip_start
            event_trip_end = vehicle.trip_end
            event_charging_end = vehicle.charging_end
            event_charging_interrupt = vehicle.charging_interrupt
            event_relocating_end = vehicle.relocating_end
            events = yield event_trip_end | event_charging_end \
                           | event_charging_interrupt | event_relocating_end
            '''if event_trip_start in events:
                for trip in self.waiting_list:
                    if trip.mode == 'unassigned':
                        self.trip_task(trip)
                        yield self.env.timeout(0.001)'''

            if event_trip_end in events:
                print(f'A vehicle get idle at {self.env.now}')
                self.env.process(self.charge_task(vehicle))
                yield self.env.timeout(0.001)
                self.relocate_task(vehicle)
                yield self.env.timeout(0.001)
                for trip in self.waiting_list:
                    if trip.mode == 'unassigned':
                        self.trip_task(trip)
                        yield self.env.timeout(0.001)
                self.env.process(self.parking_task(vehicle))
                yield self.env.timeout(0.001)

            if event_charging_end in events:
                print(f'A vehicle get charged at {self.env.now}')
                self.relocate_task(vehicle)
                for trip in self.waiting_list:
                    if trip.mode == 'unassigned':
                        self.trip_task(trip)
                        yield self.env.timeout(0.001)
                self.env.process(self.parking_task(vehicle))
                yield self.env.timeout(0.001)

            if event_charging_interrupt in events:
                print(f'Charging get interrupted at {self.env.now}')
                for trip in self.waiting_list:
                    if trip.mode == 'unassigned':
                        self.trip_task(trip)
                        yield self.env.timeout(0.001)
                self.env.process(self.parking_task(vehicle))
                yield self.env.timeout(0.001)

            if event_relocating_end in events:
                print(f'vehicle {vehicle.id} finish relocating at {self.env.now}')
                for trip in self.waiting_list:
                    if trip.mode == 'unassigned':
                        self.trip_task(trip)
                        yield self.env.timeout(0.001)
                self.env.process(self.parking_task(vehicle))
                yield self.env.timeout(0.001)

    def obs_Ve(self, vehicle):
        while True:
            t_now = self.env.now
            self.t.append(t_now)
            vehicle.info['SOC'].append(vehicle.charge_state)
            vehicle.info['location'].append([vehicle.location.lat, vehicle.location.long])
            vehicle.info['position'].append(vehicle.position)
            vehicle.info['mode'].append(vehicle.mode)
            yield self.env.timeout(1)

    def obs_CS(self, charging_station):
        while True:
            charging_station.queue.append(charging_station.plugs.count)
            yield self.env.timeout(1)

    def obs_PK(self, parking):
        while True:
            parking.queue.append(parking.capacity.count)
            yield self.env.timeout(1)

    def save_results(self, iteration):
        trips_info = []
        for i in self.waiting_list:
            trips_info.append(i.info)
        results = pd.DataFrame(trips_info)
        results_charging_demand = pd.DataFrame(self.demand_generated)
        '''results_q_table = pd.DataFrame(self.learner.q_table)
        results_q_table.to_csv('q_table.csv')'''
        with pd.ExcelWriter("results.xlsx", engine="openpyxl", mode='a') as writer:
            results.to_excel(writer, sheet_name=f'Trips{iteration}')
            results_charging_demand.to_excel(writer, sheet_name=f'demand_generated{iteration}')

            pd_ve = pd.DataFrame()
            for j in self.vehicles:
                pd_ve = pd_ve.append(pd.DataFrame([j.info["mode"]]))
            pd_ve.to_excel(writer, sheet_name=f'Vehicle_{iteration}')

            pd_cs = pd.DataFrame()
            for c in self.charging_stations:
                pd_cs = pd_cs.append([c.queue])
            pd_cs.to_excel(writer, sheet_name=f'CS_{iteration}')

            """for p in self.parkings:
                pd.DataFrame([p.queue]).to_excel(writer, sheet_name='PK_%s' % p.id)"""
