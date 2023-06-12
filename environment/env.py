import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
import sumo_rl
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import gym
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd

from .traffic_signal import TrafficSignal

from gym.utils import EzPickle, seeding


LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


# SUMO Environment for Traffic Signal Control

# :param net_file: (str) SUMO .net.xml file
# :param route_file: (str) SUMO .rou.xml file
# :param out_csv_name: (Optional[str]) name of the .csv output with simulation results. If None no output is generated
# :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
# :param virtual_display: (Optional[Tuple[int,int]]) Resolution of a virtual display for rendering
# :param begin_time: (int) The time step (in seconds) the simulation starts
# :param num_seconds: (int) Number of simulated seconds on SUMO. The time in seconds the simulation must end.
# :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
# :param delta_time: (int) Simulation seconds between actions
# :param min_green: (int) Minimum green time in a phase
# :param max_green: (int) Max green time in a phase
# :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
# :sumo_seed: (int/string) Random seed for sumo. If 'random' it uses a randomly chosen seed.
# :fixed_ts: (bool) If true, it will follow the phase configuration in the route_file and ignore the actions.
# :sumo_warnings: (bool) If False, remove SUMO warnings in the terminal

class SumoEnvironment(gym.Env):
    CONNECTION_LABEL = 0  

    def __init__(
        self, 
        net_file: str, 
        route_file: str, 
        out_csv_name: Optional[str] = None, 
        use_gui: bool = False, 
        virtual_display: Optional[Tuple[int,int]] = None,
        begin_time: int = 0, 
        num_seconds: int = 20000, 
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1, 
        delta_time: int = 5, 
        yellow_time: int = 2, 
        min_green: int = 5, 
        max_green: int = 50, 
        single_agent: bool = False, 
        sumo_seed: Union[str,int] = 'random', 
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
    ):
        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.virtual_display = virtual_display

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)
        self.ts_ids = list(conn.trafficlight.getIDList())
        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time, 
                                                  self.yellow_time, 
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  conn) for ts in self.ts_ids}
        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = EnvSpec('SUMORL-v0')
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}

  #This function starts the simulation  
    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport)]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self.virtual_display is not None:
                sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        
        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")                

#This function resets the environment 
    def reset(self):
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        self._start_simulation()

        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time, 
                                                  self.yellow_time, 
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  self.sumo) for ts in self.ts_ids}
        self.vehicles = dict()
        return self._compute_observations()[self.ts_ids[0]]
        
# Return current simulation second on SUMO
    @property
    def sim_step(self):

        return self.sumo.simulation.getTime()

#This function moves the environment to the next step
    def step(self, action):
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        self._compute_info()
        return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], dones['__all__'], {}

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

#This function sets the next green phase for the traffic signals
    def _apply_actions(self, actions):
        if self.traffic_signals[self.ts_ids[0]].time_to_act:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
    
    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones['__all__'] = self.sim_step > self.sim_max_time
        return dones
    
#This functions computes the information that is saved in the CSV file.
# Time step
# Total wait time of vehicles
# Average wait time of vehicles 
# Total number of queued cars
# Reward

    def _compute_info(self):
        info = self._compute_step_info()
        self.metrics.append(info)
        total_wait_times = [item['total_wait_time'] for item in self.metrics]
        step_times = [item['step_time'] for item in self.metrics]
        total_stopped_vehicles = [item['total_stopped'] for item in self.metrics]
        average_wait_times = [item['average_wait_time'] for item in self.metrics]

        if not hasattr(self, 'fig'):
            # Create the initial plots if they don't exist
            self.fig = go.Figure()
            self.fig.add_trace(go.Scatter(x=step_times, y=total_wait_times, mode='lines', name='Total Wait Time'))
            self.fig.update_layout(
                xaxis_title='Time Step',
                yaxis_title='Total Wait Time',
                title='Total Wait Time Per Time Step'
            )
            self.plot = st.plotly_chart(self.fig)

            self.fig2 = go.Figure()
            self.fig2.add_trace(go.Scatter(x=step_times, y=total_stopped_vehicles, mode='lines', name='Total Queued Vehicles'))
            self.fig2.update_layout(
                xaxis_title='Time Step',
                yaxis_title='Total Queued Vehicles',
                title='Total Queued Vehicles Per Time Step'
            )
            self.plot2 = st.plotly_chart(self.fig2)

            self.fig3 = go.Figure()
            self.fig3.add_trace(go.Scatter(x=step_times, y=average_wait_times, mode='lines', name='Average Wait Time'))
            self.fig3.update_layout(
                xaxis_title='Time Step',
                yaxis_title='Average Wait Time',
                title='Average Wait Time Per Time Step'
            )
            self.plot3 = st.plotly_chart(self.fig3)
        else:
            # Update the existing plots
            self.fig.data[0].x = step_times
            self.fig.data[0].y = total_wait_times
            self.fig2.data[0].x = step_times
            self.fig2.data[0].y = total_stopped_vehicles
            self.fig3.data[0].x = step_times
            self.fig3.data[0].y = average_wait_times

        # Update the plots in the Streamlit interface
        self.plot.plotly_chart(self.fig)
        self.plot2.plotly_chart(self.fig2)
        self.plot3.plotly_chart(self.fig3)

    def _compute_observations(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space
    
    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()
    
    def _compute_step_info(self):
        total_halting_vehicles = sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids)
        total_wait_time = sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids)
        average_wait_time = total_wait_time / total_halting_vehicles if total_halting_vehicles > 0 else 0.0
        
        return {
            'step_time': self.sim_step,
            'reward': self.traffic_signals[self.ts_ids[0]].last_reward,
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': total_wait_time,
            'average_wait_time': average_wait_time
        }

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        try:
            self.disp.stop()
        except AttributeError:
            pass
        self.sumo = None
    
    def __del__(self):
        self.close()
    
    def render(self, mode='human'):
        if self.virtual_display:
            img = self.disp.grab()
            if mode == 'rgb_array':
                return np.array(img)
            return img         
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_conn{}_run{}'.format(self.label, run) + '.csv', index=False)
            st.write("The Results CSV file saved")
            

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)
    