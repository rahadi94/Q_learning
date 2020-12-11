import pandas as pd
import numpy as np
import simpy
import random
from Fleet_sim.location import find_zone, closest_facility
from Fleet_sim.log import lg
from Fleet_sim.read import charging_cost
from Fleet_sim.trip import Trip
from Fleet_sim.Q_learner import RL_agent
from Fleet_sim.Matching import matching


class Model:

    def __init__(self, env, vehicles, charging_stations, zones, parkings, simulation_time, episode):
        self.t = []
        self.episode = episode
        self.parkings = parkings
        self.zones = zones
        self.charging_stations = charging_stations
        self.vehicles = vehicles
        self.trip_list = []
        self.waiting_list = []
        self.simulation_time = simulation_time
        self.env = env
        self.trip_start = env.event()
        self.demand_generated = []
        self.discharging_demand_generated = []
        self.utilization = []
        self.vehicle_id = None
        self.learner = RL_agent(env, episode=self.episode)

    def park(self, vehicle, parking):
        if self.env.now <= 5:
            vehicle.parking(parking)
            yield vehicle.parking_stop
        else:
            if vehicle.mode == 'idle':
                vehicle.mode = 'circling'
                lg.info(f'vehicle {vehicle.id} starts cruising at {self.env.now}')
                circling_interruption = vehicle.circling_stop
                vehicle.t_start_circling = self.env.now
                circling_finish = self.env.timeout(10.5)
                parking_events = yield circling_interruption | circling_finish
                if circling_interruption in parking_events:
                    # vehicle.charge_state -= (10 * 0.5 * 15 / 100 * 2)
                    lg.info(f'vehicle {vehicle.id} interrupts cruising at {self.env.now}')
                elif circling_finish in parking_events:
                    if vehicle.mode == 'circling':
                        lg.info(f'vehicle {vehicle.id} stops cruising at {self.env.now}')
                        # circling_time = self.env.now - vehicle.t_start_circling
                        # vehicle.charge_state -= (circling_time * 0.5 * 15 / 100 * 2)
                        vehicle.send_parking(parking)
                        yield self.env.timeout(vehicle.time_to_parking)
                        t_start_parking = self.env.now
                        vehicle.parking(parking)
                        yield vehicle.parking_stop
                        vehicle.reward['parking'] += (self.env.now - t_start_parking)

    def parking_task(self, vehicle):
        if vehicle.mode in ['circling', 'idle']:
            free_PL = [x for x in self.parkings if len(x.capacity.queue) == 0]
            if len(free_PL) >= 1:
                parking = closest_facility(free_PL, vehicle)
                with parking.capacity.request() as req:
                    yield req
                    yield self.env.process(self.park(vehicle, parking))
            else:
                return

    def relocate(self, vehicle, target_zone):
        vehicle.relocate(target_zone)
        target_zone.update(self.vehicles)
        yield self.env.timeout(vehicle.time_to_relocate)
        vehicle.finish_relocating(target_zone)
        vehicle.relocating_end.succeed()
        vehicle.relocating_end = self.env.event()

    def relocate_check(self, vehicle):
        for zone in self.zones:
            zone.update(self.vehicles)
        vehicle.position = find_zone(vehicle.location, self.zones)
        time = self.env.now
        for i in range(0, 24):
            if i * 60 <= time % 1440 <= (i + 1) * 60:
                hour = i
        if vehicle.charge_state >= 50 and vehicle.mode in ['idle', 'parking'] and len(
                vehicle.position.list_of_vehicles) >= vehicle.position.demand.iloc[0, hour]:
            return True

    def relocate_task(self, vehicle):
        time = self.env.now
        for i in range(0, 24):
            if i * 60 <= time % 1440 <= (i + 1) * 60:
                hour = i
        target_zones = [z for z in self.zones if len(z.list_of_vehicles) <= z.demand.iloc[0, hour]]
        if len(target_zones) > 1:
            target_zone = closest_facility(target_zones, vehicle)
            if vehicle.mode == 'parking':
                vehicle.parking_stop.succeed()
                vehicle.parking_stop = self.env.event()
            self.env.process(self.relocate(vehicle, target_zone))

    def start_charge(self, charging_station, vehicle):
        vehicle.send_charge(charging_station)
        vehicle.charging_demand = dict(vehicle_id=vehicle.id, time_send=self.env.now,
                                       time_enter=self.env.now + vehicle.time_to_CS,
                                       time_start=None, SOC_end=None,
                                       SOC_send=vehicle.charge_state, lat=vehicle.location.lat,
                                       long=vehicle.location.long,
                                       v_hex=vehicle.position.hexagon,
                                       CS_location=[charging_station.location.lat, charging_station.location.long],
                                       v_position=vehicle.position.id, CS_position=charging_station.id,
                                       distance=vehicle.location.distance(charging_station.location)[0])
        yield self.env.timeout(vehicle.time_to_CS)
        vehicle.charging(charging_station)
        vehicle.t_arriving_CS = self.env.now

    def finish_charge(self, charging_station, vehicle):
        try:
            yield self.env.timeout(vehicle.charge_duration)
            vehicle.finish_charging(charging_station)
            vehicle.charging_demand['SOC_end'] = vehicle.charge_state
            vehicle.charging_end.succeed()
            vehicle.charging_end = self.env.event()
        except simpy.Interrupt:
            old_SOC = vehicle.charge_state
            vehicle.charge_state += float((charging_station.power * (float(self.env.now) - vehicle.t_start_charging)) \
                                          / (vehicle.battery_capacity / 100))
            vehicle.charging_demand['SOC_end'] = vehicle.charge_state
            vehicle.mode = 'idle'
            vehicle.charging_interrupt.succeed()
            vehicle.charging_interrupt = self.env.event()
            for j in range(0, 24):
                if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                    h = j
            vehicle.reward['charging'] += (vehicle.charge_state - old_SOC) / 100 * 50 * charging_cost[h] / 100
            if isinstance(vehicle.reward['charging'], np.ndarray):
                vehicle.reward['charging'] = vehicle.reward['charging'][0]
            vehicle.costs['charging'] += (vehicle.charging_threshold - vehicle.charge_state) * \
                                         charging_cost[h] / 100
            lg.info(f'Warning!!!Charging state of vehicle {vehicle.id} is {vehicle.charge_state} at {self.env.now} ')
        self.demand_generated.append(vehicle.charging_demand)

    def charge_check(self, vehicle):
        if self.env.now > 10 and vehicle.mode in ['idle', 'parking']:
            if vehicle.charging_count > 0:
                self.learner.update_value(vehicle, self.charging_stations, self.vehicles, self.waiting_list)
            action = self.learner.take_action(vehicle, self.charging_stations, self.vehicles, self.waiting_list)
            vehicle.action = action
            vehicle.charging_count += 1
            return action

    def charge_task(self, vehicle, action):
        if action == 0:
            charging_station = closest_facility(self.charging_stations, vehicle)
        elif action == 1:
            free_CS = [x for x in self.charging_stations if len(x.plugs.queue) == 0]
            if len(free_CS) >= 1:
                charging_station = closest_facility(free_CS, vehicle)
            else:
                charging_station = closest_facility(self.charging_stations, vehicle)
        elif action == 2:
            fast_CS = [x for x in self.charging_stations if x.power == 50 / 60]
            charging_station = closest_facility(fast_CS, vehicle)

        yield self.env.process(self.start_charge(charging_station, vehicle))
        prio = int((vehicle.charge_state - vehicle.charge_state % 10) / 10)
        if isinstance(prio, np.ndarray):
            prio = prio[0]
        req = charging_station.plugs.request(priority=prio)
        vehicle.mode = 'queue'
        events = yield req | vehicle.queue_interruption

        if req in events:
            vehicle.charging_demand['time_start'] = self.env.now
            lg.info(f'Vehicle {vehicle.id} starts charging at {self.env.now}')
            vehicle.t_start_charging = self.env.now
            vehicle.reward['queue'] += (vehicle.t_start_charging - vehicle.t_arriving_CS)
            if isinstance(vehicle.reward['queue'], np.ndarray):
                vehicle.reward['queue'] = vehicle.reward['queue'][0]
            vehicle.mode = 'charging'
            charging = self.env.process(self.finish_charge(charging_station, vehicle))
            yield charging | vehicle.charging_interruption
            charging_station.plugs.release(req)
            req.cancel()

            if not charging.triggered:
                charging.interrupt()
                lg.info(f'Vehicle {vehicle.id} stops charging at {self.env.now}')
                return

        else:
            vehicle.reward['queue'] += (self.env.now - vehicle.t_arriving_CS)
            if isinstance(vehicle.reward['queue'], np.ndarray):
                vehicle.reward['queue'] = vehicle.reward['queue'][0]
            vehicle.charging_demand['SOC_end'] = vehicle.charge_state
            lg.info(f'vehicle {vehicle.id} interrupts the queue')
            req.cancel()
            charging_station.plugs.release(req)
            return

    def discharging(self, vehicle, charging_station):
        vehicle.discharging(charging_station)
        vehicle.discharging_demand = dict(vehicle_id=vehicle.id, time_send=self.env.now,
                                          time_enter=self.env.now + vehicle.time_to_CS,
                                          time_start=None, SOC_end=None,
                                          SOC_send=vehicle.charge_state,
                                          v_hex=vehicle.position.hexagon,
                                          v_position=vehicle.position.id, CS_position=charging_station.id,
                                          distance=vehicle.location.distance(charging_station.location)[0])
        yield self.env.timeout(vehicle.discharge_duration)
        vehicle.finish_discharging(charging_station)
        vehicle.discharging_end.succeed()
        vehicle.discharging_end = self.env.event()

    def discharge_task(self, vehicle):
        if vehicle.charge_state > 70:
            free_CS = [x for x in self.charging_stations if len(x.plugs.queue) == 0]
            if len(free_CS) >= 1:
                charging_station = closest_facility(free_CS, vehicle)
            else:
                charging_station = closest_facility(self.charging_stations, vehicle)
            vehicle.send_charge(charging_station)
            lg.info(f'vehicle {vehicle.id} is sent for discharging')
            yield self.env.timeout(vehicle.time_to_CS)
            req = charging_station.plugs.request(priority=5)
            vehicle.mode = 'queue'
            events = yield req | vehicle.queue_interruption
            if req in events:
                vehicle.mode = 'discharging'
                lg.info(f'Vehicle {vehicle.id} starts discharging at {self.env.now}')
                yield self.env.process(self.discharging(vehicle=vehicle, charging_station=charging_station))
                charging_station.plugs.release(req)
                req.cancel()
                vehicle.discharging_demand['SOC_end'] = vehicle.charge_state
                self.discharging_demand_generated.append(vehicle.charging_demand)
            else:
                lg.info(f'vehicle {vehicle.id} interrupts the queue')
                req.cancel()
                charging_station.plugs.release(req)
                return

        else:
            return

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
        self.trip_list.append(trip)
        trip.info['mode'] = 'finished'
        vehicle.reward['revenue'] += max(((trip.distance * 1.11 + trip.duration * 0.31) + 2), 5) - \
                        float(trip.info['waiting_time']) * 0.10
        if isinstance(vehicle.reward['revenue'], np.ndarray):
            vehicle.reward['revenue'] = vehicle.reward['revenue'][0]

    def trip_task(self):
        vehicles = [x for x in self.vehicles if x.mode in ['idle', 'parking', 'circling', 'queue']]
        trips = [x for x in self.waiting_list if x.mode == 'unassigned']
        pairs = matching(vehicles, trips)
        if len(pairs) == 0:
            return
        for i in pairs:
            vehicle = i['vehicle']
            trip = i['trip']
            if vehicle.mode == 'parking':
                vehicle.parking_stop.succeed()
                vehicle.parking_stop = self.env.event()
            elif vehicle.mode == 'circling':
                vehicle.circling_stop.succeed()
                vehicle.circling_stop = self.env.event()
            elif vehicle.mode == 'queue':
                vehicle.queue_interruption.succeed()
                vehicle.queue_interruption = self.env.event()
            self.env.process(self.take_trip(trip, vehicle))
            # yield self.env.timeout(0.001)

    def trip_generation(self, zone):
        j = 0
        while True:
            j += 1
            trip = Trip(self.env, (j, zone.id), zone)
            yield self.env.timeout(trip.interarrival)
            self.trip_start.succeed()
            self.trip_start = self.env.event()
            self.trip = trip
            trip.info['arrival_time'] = self.env.now
            self.waiting_list.append(trip)
            lg.info(f'Trip {trip.id} is received at {self.env.now}')
            trip.start_time = self.env.now

    def missed_trip(self):
        while True:
            for trip in self.waiting_list:
                if trip.mode == 'unassigned' and self.env.now > (trip.start_time + 3):
                    r = random.uniform(0, 1)
                    if r < 0.1:
                        trip.mode = 'missed'
                        trip.info['mode'] = 'missed'
                        self.trip_list.append(trip)
                        self.waiting_list.remove(trip)
                        lg.info(f'trip {trip.id} is missed at {self.env.now}')
                elif trip.mode == 'unassigned' and self.env.now > (trip.start_time + 5):
                    r = random.uniform(0, 1)
                    if r < 0.5:
                        trip.mode = 'missed'
                        trip.info['mode'] = 'missed'
                        self.trip_list.append(trip)
                        self.waiting_list.remove(trip)
                        lg.info(f'trip {trip.id} is missed at {self.env.now}')
                elif trip.mode == 'unassigned' and self.env.now > (trip.start_time + 10):
                    trip.mode = 'missed'
                    trip.info['mode'] = 'missed'
                    self.trip_list.append(trip)
                    self.waiting_list.remove(trip)
                    lg.info(f'trip {trip.id} is missed at {self.env.now}')
                if trip.mode == 'missed':
                    vehicle_responsible = [x for x in self.vehicles if x.distance_1(trip) <= 3 and
                                           x.mode in ['charging', 'discharging', 'queue', 'ertc']]
                    for vehicle in vehicle_responsible:
                        vehicle.reward['missed'] += 10
            yield self.env.timeout(1)

    def hourly_charging_relocating(self):
        while True:
            yield self.env.timeout(60)
            for vehicle in self.vehicles:
                if vehicle.mode == 'parking':
                    action = self.charge_check(vehicle)
                    if action in [0, 1, 2]:
                        self.env.process(self.charge_task(vehicle, action))
                        yield self.env.timeout(0.001)
                    elif action == 3:
                        self.env.process(self.discharge_task(vehicle))
                        yield self.env.timeout(0.001)
                    elif self.relocate_check(vehicle):
                        self.relocate_task(vehicle)
                        yield self.env.timeout(0.001)

    def charging_interruption(self):
        while True:
            for vehicle in self.vehicles:
                if vehicle.mode in ['charging']:
                    vehicle.position = find_zone(vehicle.location, self.zones)
                    vehicle.position.update(self.vehicles)
                    power = 11 / 60
                    try:
                        duration = self.env.now - vehicle.t_start_charging
                    except:
                        duration = 0
                    soc = vehicle.charge_state + ((power * duration) / (vehicle.battery_capacity / 100))
                    supply = len([v for v in self.vehicles if v.location.distance_1(vehicle.location) <= 5
                                  and v.charge_state >= 30 and
                                  vehicle.mode in ['idle', 'parking', 'circling', 'queue']])
                    wl = len([t for t in self.waiting_list if t.origin.distance(vehicle.location)[0] <= 5])
                    if soc > 70 and wl >= supply:
                        vehicle.charging_interruption.succeed()
                        vehicle.charging_interruption = self.env.event()
                    yield self.env.timeout(0.001)
                yield self.env.timeout(15)

    def run(self):
        while True:
            yield self.env.timeout(2)
            if len(self.waiting_list) >= 1:
                self.trip_task()

    def run_vehicle(self, vehicle):
        while True:
            if self.env.now == 0:
                self.env.process(self.parking_task(vehicle))

            event_trip_end = vehicle.trip_end
            event_charging_end = vehicle.charging_end
            event_charging_interrupt = vehicle.charging_interrupt
            event_relocating_end = vehicle.relocating_end
            event_discharging_end = vehicle.discharging_end
            events = yield event_trip_end | event_charging_end \
                           | event_charging_interrupt | event_relocating_end | event_discharging_end

            if event_trip_end in events:
                lg.info(f'A vehicle gets idle at {self.env.now}')
                action = self.charge_check(vehicle)
                if action in [0, 1, 2]:
                    self.env.process(self.charge_task(vehicle, action))
                    yield self.env.timeout(0.001)
                elif action == 3:
                    self.env.process(self.discharge_task(vehicle))
                    yield self.env.timeout(0.001)
                else:
                    if self.relocate_check(vehicle):
                        self.relocate_task(vehicle)
                        yield self.env.timeout(0.001)
                    else:
                        self.env.process(self.parking_task(vehicle))
                        yield self.env.timeout(0.001)

            if event_charging_end in events:
                lg.info(f'A vehicle gets charged at {self.env.now}')
                if self.relocate_check(vehicle):
                    self.relocate_task(vehicle)
                    yield self.env.timeout(0.001)
                else:
                    self.env.process(self.parking_task(vehicle))
                    yield self.env.timeout(0.001)

            if event_discharging_end in events:
                lg.info(f'A vehicle gets discharged at {self.env.now}')
                if self.relocate_check(vehicle):
                    self.relocate_task(vehicle)
                    yield self.env.timeout(0.001)
                else:
                    self.env.process(self.parking_task(vehicle))
                    yield self.env.timeout(0.001)

            if event_charging_interrupt in events:
                lg.info(f'Charging gets interrupted at {self.env.now}')
                self.env.process(self.parking_task(vehicle))
                yield self.env.timeout(0.001)

            if event_relocating_end in events:
                lg.info(f'vehicle {vehicle.id} finishes relocating at {self.env.now}')
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
            charging_station.queue.append([charging_station.plugs.count, len(charging_station.plugs.queue)])
            yield self.env.timeout(1)

    def obs_PK(self, parking):
        while True:
            parking.queue.append(parking.capacity.count)
            yield self.env.timeout(1)

    def save_results(self, episode):
        trips_info = []
        for i in self.trip_list:
            trips_info.append(i.info)
        results = pd.DataFrame(trips_info)
        results_charging_demand = pd.DataFrame(self.demand_generated)
        results_discharging_demand = pd.DataFrame(self.discharging_demand_generated)
        self.learner.q_table.to_csv('q_table.csv')
        # with pd.ExcelWriter("results.xlsx", engine="openpyxl", mode='a') as writer:
        results.to_csv(f'results/trips{episode}.csv')
        results_charging_demand.to_csv(f'results/charging{episode}.csv')
        results_discharging_demand.to_csv(f'results/discharging{episode}.csv')

        pd_ve = pd.DataFrame()
        pd_reward = pd.DataFrame()
        for j in self.vehicles:
            pd_ve = pd_ve.append(pd.DataFrame([j.info["mode"], j.info['SOC'], j.info['location']]))
            pd_reward = pd_reward.append(pd.DataFrame([j.total_rewards['state'], j.total_rewards['action'],
                                                       j.total_rewards['reward']]))
        pd_ve.to_csv(f'results/vehicles{episode}.csv')
        pd_reward.to_csv(f'results/rewards{episode}.csv')

        pd_cs = pd.DataFrame()
        for c in self.charging_stations:
            pd_cs = pd_cs.append([c.queue])
        pd_cs.to_csv(f'results/CSs{episode}.csv')

        """for p in self.parkings:
            pd.DataFrame([p.queue]).to_excel(writer, sheet_name='PK_%s' % p.id)"""
