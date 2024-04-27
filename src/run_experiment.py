import csv
import nengo
import carla
import math
import numpy as np
import random
import argparse
from nengo_ex.synapses import IdealDelay, IdealDelayLowpass
from nengo_ex.node import ControllableNode
from environment.car import CarlaInterface, ACTION_SIZE, STATE_SIZE, bicycle_model_dynamics_center, bicycle_model_state_center
from networks.mpc import Predictor
from scipy.optimize import minimize

argparser = argparse.ArgumentParser(description="Run MPC on Carla")
argparser.add_argument("--host", default="localhost", help="IP address of the CARLA Simulator host (default: localhost)")
argparser.add_argument("--port", default=2000, type=int, help="TCP port of the CARLA Simulator host (default: 2000)")
argparser.add_argument("--car", default="vehicle.tesla.model3", help="Car model blueprint (default: vehicle.tesla.model3)")
argparser.add_argument("--map", default="Town04", help="Map to use (default: Town04)")
argparser.add_argument("--waypoints", default="1,369,55,320,323,0", help="Waypoints to use as list of spawn points ids (default: 1,369,55,320,323,0)")
argparser.add_argument("--predict-dt", default=0.1, type=float, help="Prediction time step (default: 0.1)")
argparser.add_argument("--predict-horizon", default=5, type=int, help="Prediction horizon (default: 5)")
argparser.add_argument("--synapse", default=0.05, type=float, help="Synapse time constant (default: 0.05)")
argparser.add_argument("--look-ahead", default=60, type=int, help="Look ahead distance in meters (default: 60)")
argparser.add_argument("--road-degree", default=3, type=int, help="Degree of polynomial to fit to road (default: 3)")
argparser.add_argument("--learning-rate", default=1e-4, type=float, help="Learning rate (default: 1e-4)")
argparser.add_argument("--predictor-neurons", default=1000, type=int, help="Number of neurons in adaptive predictor (default: 1000)")
argparser.add_argument("--steering-malfunction", default=0.25, type=float, help="Steering malfunction (default: 0.25)")
argparser.add_argument("--simulation-time", default=60, type=int, help="Simulation time in seconds (default: 60)")
argparser.add_argument("--waypoints-resolution", default=1, type=int, help="Waypoints resolution in meters (default: 1)")
argparser.add_argument("--swift", default=0, type=int, help="Swift malfunction direction every n seconds. 0 or negative to disable (default: 0)")
argparser.add_argument("--vehicle-wheelbase", type=float, default=2.3, help="Vehicle's wheelbase length (default: 2.3)")

args = argparser.parse_args()

np.random.seed(0)
random.seed(0)

# Simulation parameters
DT = 0.001
PREDICT_DT = args.predict_dt
PREDICTION_HORIZON = args.predict_horizon
PREDICT_SIM_TIME = PREDICT_DT

# Adaptive predictor parameters
MAX_STAT = [500., 500., 1., 1., 40., 40., 40., 40., 90.,]

SYNAPSE = args.synapse
LEARNING_RATE = args.learning_rate
PREDICTOR_NEURONS = args.predictor_neurons
LOCAL_STATE_SIZE = 5
DYNAMICS_SIZE = STATE_SIZE - 1 # use yaw as angle

# Controls parameters
LOOK_AHEAD = args.look_ahead
ROAD_DEGREE = args.road_degree
V_REF = 100 # km/h
VEHICLE_WHEELBASE = args.vehicle_wheelbase

def format_e(n):
    a = '%E' % n
    split = a.split('E')

    mantissa = split[0].rstrip('0').rstrip('.')
    exponent = split[1].lstrip('+').lstrip('-').lstrip('0').rstrip('0')
    exponent_sign = '-' if split[1].startswith('-') else ''

    return f'{mantissa}E{exponent_sign}{exponent}'

LOG_FILE_NAME = "state_log_{}_{}_{}_{}_{}.csv".format(
    PREDICTION_HORIZON,
    PREDICT_DT,
    args.car.split('.')[-1],
    "norm" if args.steering_malfunction == 0 else f"steermal={args.steering_malfunction}",
    "bicycle" if PREDICTOR_NEURONS == 0 else "adaptive_{}_{}_{}".format(PREDICTOR_NEURONS, SYNAPSE, format_e(LEARNING_RATE)))

csv_file = open(LOG_FILE_NAME, mode="w", newline='')
log_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
log_writer.writerow(["px", "py", "yaw_cos", "yaw_sin", "vx", "vy", "ax", "ay", "vr", "dx_err", "dy_err", "dyaw_err", "dvx_err", "dvy_err", "dax_err", "day_err", "dvr_err", "predict_cost"])

carla_env = CarlaInterface(default_dt=DT, host=args.host, port=args.port)

def bicycle(x):
    throttle, steer, vx, vy, ax, ay, vr = x
    new_x = [throttle, steer] + list(np.multiply([1, 0, vx, vy, ax, ay, vr], MAX_STAT[2:]))
    return np.array(bicycle_model_dynamics_center(new_x, vehicle_wheelbase=VEHICLE_WHEELBASE)) * PREDICT_DT

def localize(t, x):
    """
    Convert the state of the vehicle in the global coordinate system to the local coordinate system.
    In the local coordinate system, the vehicle is always at the origin and the yaw is always 0.

    Args:
        t (float): time (not used)
        x (np.array): state of the vehicle in the global coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)

    Returns:
        np.array: state of the vehicle in the local coordinate system (vx, vy, ax, ay, vr)
    """
    [_, _, yaw_cos, yaw_sin, vx, vy, ax, ay, vr] = x
    vehicle_yaw = math.atan2(yaw_sin, yaw_cos)
    R_INV = np.array([[np.cos(-vehicle_yaw), -np.sin(-vehicle_yaw)],
                        [np.sin(-vehicle_yaw), np.cos(-vehicle_yaw)]])
                        
    v_oriented = np.matmul(R_INV, np.array([vx, vy]).T).T
    a_oriented = np.matmul(R_INV, np.array([ax, ay]).T).T
    
    return np.concatenate([
        v_oriented,
        a_oriented,
        [vr]
    ])

def globalize(t, x):
    """
    Convert the dynamics of the vehicle in the local coordinate system to the global coordinate system.

    Args:
        t (float): time (not used)
        x (np.array): dynamics of the vehicle in the local coordinate system (dx, dy, dyaw, dvx, dvy, dax, day, dvr, oyaw_cos, oyaw_sin)

    Returns:
        np.array: dynamics of the vehicle in the global coordinate system (dx, dy, dyaw, dvx, dvy, dax, day, dvr)
    """
    [dx, dy, dyaw, dvx, dvy, dax, day, dvr, oyaw_cos, oyaw_sin] = x 
    vehicle_yaw = math.atan2(oyaw_sin, oyaw_cos)
    R = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw)],
                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw)]])
    
    p_oriented = np.matmul(R, np.array([dx, dy]).T).T            
    v_oriented = np.matmul(R, np.array([dvx, dvy]).T).T
    a_oriented = np.matmul(R, np.array([dax, day]).T).T
    
    return np.concatenate([
        p_oriented,
        [dyaw],
        v_oriented,
        a_oriented,
        [dvr]
    ])

def calc_state(t, x):
    """
    Calculate the next state of the vehicle based on the current state and the dynamics.

    Args:
        t (float): time (not used)
        x (np.array): state and dynamics of the vehicle in the local coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr, dx, dy, dyaw, dvx, dvy, dax, day, dvr)

    Returns:
        np.array: next state of the vehicle in the local coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
    """
    [px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr, dx, dy, dyaw, dvx, dvy, dax, day, dvr] = x

    yaw = math.atan2(yaw_sin, yaw_cos)

    px += dx
    py += dy
    yaw += dyaw
    vx += dvx
    vy += dvy
    ax += dax
    ay += day
    vr += dvr
    
    yaw_cos = math.cos(yaw)
    yaw_sin = math.sin(yaw)
    
    return [px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr]

# Build the predictive model
p_model = nengo.Network(seed=0)
with p_model:
    p_controller = ControllableNode(size_out=ACTION_SIZE)
        
    p_env_local = ControllableNode(size_out=LOCAL_STATE_SIZE)
        
    p_predictor = Predictor(
        action_size=ACTION_SIZE,
        state_size=LOCAL_STATE_SIZE,
        dynamics_size=DYNAMICS_SIZE,
        static_predictor_n_neurons=3000,
        static_predictor_function=bicycle,
        adaptive_predictor_n_neurons=PREDICTOR_NEURONS,
        adaptive_predictor_state_slice=slice(0, None),
        synapse_tau=SYNAPSE,
        trainable=False,
        adaptive_predictor_weights=[[0] * PREDICTOR_NEURONS] * DYNAMICS_SIZE,
        adaptive_predictor_neuron_type=nengo.LIF(),
        static_predictor_neuron_type=nengo.Direct(),
        pes_pre_synapse=IdealDelayLowpass(SYNAPSE, PREDICT_DT),
    )
    
    nengo.Connection(p_env_local, p_predictor.input_state, transform=np.reciprocal(MAX_STAT[-LOCAL_STATE_SIZE:]), synapse=None)
    nengo.Connection(p_controller, p_predictor.input_action, synapse=None)
    
    p_pred_dynamics_probe = nengo.Probe(p_predictor.output, sample_every=PREDICT_SIM_TIME)

predict_sim = nengo.Simulator(p_model, DT, progress_bar=False)

def predict_states_neurons(car_state, actions, predictor_weights):
    """
    Predict the states of the vehicle for the given actions and initial state using the neuromorphic predictor model.

    Args:
        car_state (np.array): initial state of the vehicle in the global coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
        actions (np.array): actions to be applied to the vehicle (throttle, steering)
        predictor_weights (np.array): weights of the adaptive ensemble of the predictor

    Returns:
        list: predicted states of the vehicle in the global coordinate system for each prediction step (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
    """

    predict_sim.reset()
    if predictor_weights is not None:
        predict_sim.signals[
            predict_sim.model.sig[
                p_predictor.adaptive_connection]["weights"]] = predictor_weights

    global_state = car_state
    states = []
    for i in range(PREDICTION_HORIZON):
        local_state = localize(0, global_state)

        p_controller.value = actions[i * ACTION_SIZE:(i + 1) * ACTION_SIZE]
        p_env_local.value = local_state

        predict_sim.run(PREDICT_SIM_TIME)

        global_dynamics = globalize(0, np.concatenate((predict_sim.data[p_pred_dynamics_probe][-1], global_state[2:4])))
        global_state = calc_state(0, np.concatenate((global_state, global_dynamics)))
        states.append(global_state)

    return states

def predict_states_bicycle(car_state, actions):
    """
    Predict the states of the vehicle for the given actions and initial state using the bicycle model.

    Args:
        car_state (np.array): initial state of the vehicle in the global coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
        actions (np.array): actions to be applied to the vehicle (throttle, steering)

    Returns:
        list: predicted states of the vehicle in the global coordinate system for each prediction step (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
    """

    throttles = actions[::2]
    steers = actions[1::2]

    states = []
    for i in range(len(throttles)):
        throttle = throttles[i]
        steer = steers[i]

        car_state = bicycle_model_state_center(car_state, [throttle, steer], PREDICT_DT)
        states.append(car_state)

    return states

def predict_states(car_state, actions, predictor_weights):
    """
    Predict the states of the vehicle for the given actions and initial state using the specified model.
    
    Args:
        car_state (np.array): initial state of the vehicle in the global coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
        actions (np.array): actions to be applied to the vehicle (throttle, steering)
        predictor_weights (np.array): weights of the adaptive ensemble of the predictor

    Returns:
        list: predicted states of the vehicle in the global coordinate system for each prediction step (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
    """

    if PREDICTOR_NEURONS == 0:
        return predict_states_bicycle(car_state, actions)
    else:
        return predict_states_neurons(car_state, actions, predictor_weights)

def state_cost(u, du, state, road_coeffs):
    """
    Calculate the cost of the given state and action.

    Args:
        u (np.array): action to be applied to the vehicle (throttle, steering)
        du (np.array): change in action to be applied to the vehicle (throttle, steering)
        state (np.array): state of the vehicle in the global coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
        road_coeffs (np.array): coefficients of the road polynomial

    Returns:
        float: cost of the given state and action
    """

    px, py, yaw_cos, yaw_sin, vx, vy, _, _, _ = state
    throttle, steer = u
    dthrottle, dsteer = du

    v = np.hypot(vx, vy) * 3.6  # m/s to km/h
    yaw = np.arctan2(yaw_sin, yaw_cos)

    cte = (np.polyval(road_coeffs, px) - py) / 3  # normalize by 3 meters
    road_heading = np.arctan(np.polyval(np.polyder(road_coeffs), px))
    eyaw = (yaw - road_heading) / (np.pi / 4)  # normalize by 45 degrees

    v_err = (v - V_REF)

    return (
        550 * cte ** 2 # cte
        + 200 * eyaw ** 2 # eyaw
        + 1 * v_err ** 2 # speed
        + 110 * throttle ** 2 # throttle
        + 250 * steer ** 2 # steer
        + 0 * (dthrottle) ** 2 # throttle change
        + 50 * (dsteer) ** 2 # steer change
    )

def cost(u, car_state, road_coeffs, predictor_weights, prev_action):
    """
    Calculate the cost of the given action sequence.

    Args:
        u (np.array): action sequence to be applied to the vehicle (throttle, steering)
        car_state (np.array): initial state of the vehicle in the global coordinate system (px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr)
        road_coeffs (np.array): coefficients of the road polynomial
        predictor_weights (np.array): weights of the adaptive ensemble of the predictor
        prev_action (np.array): previous action applied to the vehicle (throttle, steering)

    Returns:
        float: cost of the given action sequence
    """

    _, _, yaw_cos, yaw_sin, vx, vy, ax, ay, vr = car_state

    vehicle_yaw = np.arctan2(yaw_sin, yaw_cos)

    R_INV = np.array([[np.cos(-vehicle_yaw), -np.sin(-vehicle_yaw)],
                          [np.sin(-vehicle_yaw), np.cos(-vehicle_yaw)]])
    
    v_oriented = np.matmul(R_INV, np.array([vx, vy]).T).T
    a_oriented = np.matmul(R_INV, np.array([ax, ay]).T).T

    car_state = [
        0,
        0,
        1,
        0,
        v_oriented[0],
        v_oriented[1],
        a_oriented[0],
        a_oriented[1],
        vr,
    ]
    states = predict_states(car_state, u, predictor_weights)

    cost = 0
    for i, state in enumerate(states):
        action = u[i * ACTION_SIZE:(i + 1) * ACTION_SIZE]
        last_action = u[(i - 1) * ACTION_SIZE:i * ACTION_SIZE] if i > 0 else prev_action
        daction = action - last_action
        cost += state_cost(action, daction, state, road_coeffs)

    return cost

def control_logic(t):
    """
    Control logic of the vehicle. It is called at each simulation step and returns the throttle and steering to be applied to the vehicle.
    New actions are calculated every PREDICT_DT seconds, and the same action is applied to the vehicle during this time.

    Args:
        t (float): current simulation time

    Returns:
        tuple: throttle and steering to be applied to the vehicle and the last cost of the MPC (throttle, steering, cost)
    """

    vehicle_location = carla_env.vehicle.get_transform().location
    vehicle_yaw = np.deg2rad(carla_env.vehicle.get_transform().rotation.yaw)
    
    R = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw)],
                  [np.sin(vehicle_yaw), np.cos(vehicle_yaw)]])
    
    
    road_coeffs = carla_env.get_road_coeff(look_distance=LOOK_AHEAD, degree=ROAD_DEGREE)
    for i in np.linspace(0, LOOK_AHEAD, 100):
        pt = np.polyval(road_coeffs, i)
        pt = np.matmul(R, np.array([i, pt]).T).T
        carla_env.plot_point(
            carla.Location(x=pt[0], y=pt[1], z=2) + vehicle_location,
            2 / 60,
            0.1,
            carla.Color(r=0, g=0, b=255),
        )

    if PREDICTOR_NEURONS > 0:
        predictor_weights = run_sim.signals[
            run_sim.model.sig[
                predictor.adaptive_connection]["weights"]]
    else:
        predictor_weights = None
            
    if t > control_logic.last_control_t + PREDICT_DT:
        car_state = carla_env.get_state()

        u = minimize(
            cost,
            control_logic.last_control,
            bounds=[(-1, 1)] * ACTION_SIZE * PREDICTION_HORIZON,
            args=(car_state,
                  road_coeffs,
                  predictor_weights,
                  control_logic.last_control[:ACTION_SIZE]),
            method='SLSQP',
            options={'maxiter': 100, 'disp': True},
        )
        selected_actions = u.x

        print()
        print(u.x, u.fun, u.nit)

        states = predict_states_neurons(car_state, selected_actions, predictor_weights)
        for state in states:
            x = state[0]
            y = state[1]
            carla_env.plot_point(
                carla.Location(x=x, y=y, z=2.5 + vehicle_location.z),
                PREDICT_DT + 2 / 60,
                0.1,
                carla.Color(r=255, g=255, b=255),
            )

        control_logic.last_control = selected_actions
        control_logic.last_control_t = t
        control_logic.last_control_cost = u.fun
        
        return np.append(selected_actions[:2], u.fun)
    return np.append(control_logic.last_control[:2], control_logic.last_control_cost)
control_logic.last_control = [0] * ACTION_SIZE * PREDICTION_HORIZON
control_logic.last_control_t = -1
control_logic.last_control_cost = 0
        
def plot_factory(color, z=2.1):
    """
    Factory function for creating plotting functions for the simulator.

    Args:
        color (carla.Color): color of the plotted points
        z (float, optional): z coordinate of the plotted points. Defaults to 2.1.
    
    Returns:
        function: plotting function for the simulator
    """

    def plot_pred(t, x):
        if carla_env.is_ready:
            [px, py] = x
            carla_env.plot_point(
                carla.Location(x=px, y=py) + carla.Location(z=carla_env.vehicle.get_location().z+z),
                color=color,
                life_time=10)
    return plot_pred


# Build the main network model
model = nengo.Network(seed=0)
with model:
    env = nengo.Node(carla_env, size_in=ACTION_SIZE, size_out=STATE_SIZE)

    controller = nengo.Node(control_logic, size_out=ACTION_SIZE + 1)
    nengo.Connection(controller[:ACTION_SIZE], env, synapse=None)

    unexpected_dynamics = nengo.Node(lambda t: [0, args.steering_malfunction] if args.swift <= 0 or (t // args.swift) % 2 == 0 else [0, -args.steering_malfunction] , size_out=ACTION_SIZE)
    nengo.Connection(unexpected_dynamics, env, synapse=None)

    env_local = nengo.Node(localize, size_in=STATE_SIZE, size_out=LOCAL_STATE_SIZE)
    nengo.Connection(env, env_local, synapse=None)

    predictor = Predictor(
        action_size=ACTION_SIZE,
        state_size=LOCAL_STATE_SIZE,
        dynamics_size=DYNAMICS_SIZE,
        static_predictor_n_neurons=3000,
        static_predictor_function=bicycle,
        adaptive_predictor_n_neurons=PREDICTOR_NEURONS,
        adaptive_predictor_state_slice=slice(0, None),
        synapse_tau=SYNAPSE,
        trainable=True,
        learning_rate=LEARNING_RATE,
        adaptive_predictor_neuron_type=nengo.LIF(),
        static_predictor_neuron_type=nengo.Direct(),
        pes_pre_synapse=IdealDelayLowpass(SYNAPSE, PREDICT_DT),
    )
    
    nengo.Connection(env_local, predictor.input_state, transform=np.reciprocal(MAX_STAT[-LOCAL_STATE_SIZE:]), synapse=None)

    next_state = nengo.Node(calc_state, size_in=STATE_SIZE + DYNAMICS_SIZE)
    nengo.Connection(env, next_state[:STATE_SIZE], synapse=None)

    prediction_plotter = nengo.Node(plot_factory(carla.Color(r=255, g=0, b=0)), size_in=2, label="red")
    nengo.Connection(next_state[:2], prediction_plotter, synapse=None)

    pos_plotter = nengo.Node(plot_factory(carla.Color(r=0, g=255, b=0), z=2.2), size_in=2, label="green")
    nengo.Connection(env[:2], pos_plotter, synapse=None)

    nengo.Connection(controller[:ACTION_SIZE], predictor.input_action, synapse=None)

    global_dynamics = nengo.Node(globalize, size_in=DYNAMICS_SIZE+2, size_out=DYNAMICS_SIZE)

    nengo.Connection(predictor.output, global_dynamics[:DYNAMICS_SIZE], synapse=None)
    nengo.Connection(global_dynamics, next_state[STATE_SIZE:], synapse=None)
    nengo.Connection(env[2:4], global_dynamics[DYNAMICS_SIZE:], synapse=None)
    
    def calc_dynamics(t, x):
        [
            px1, py1, yaw_cos1, yaw_sin1, vx1, vy1, ax1, ay1, vr1,
            px2, py2, yaw_cos2, yaw_sin2, vx2, vy2, ax2, ay2, vr2,
        ] = x
        
        yaw1 = math.atan2(yaw_sin1, yaw_cos1)
        yaw2 = math.atan2(yaw_sin2, yaw_cos2)
        
        dyaw = yaw2 - yaw1
        normalized_dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        
        R_INV = np.array([[np.cos(-yaw1), -np.sin(-yaw1)],
                    [np.sin(-yaw1), np.cos(-yaw1)]])
        
        p_oriented = np.matmul(R_INV, np.array([px2-px1, py2-py1]).T).T            
        v_oriented = np.matmul(R_INV, np.array([vx2-vx1, vy2-vy1]).T).T
        a_oriented = np.matmul(R_INV, np.array([ax2-ax1, ay2-ay1]).T).T

        return [
            p_oriented[0],
            p_oriented[1],
            normalized_dyaw,
            v_oriented[0],
            v_oriented[1],
            a_oriented[0],
            a_oriented[1],
            vr2-vr1,
        ]
    real_local_dynamics = nengo.Node(calc_dynamics, size_in=STATE_SIZE*2, size_out=DYNAMICS_SIZE)
    nengo.Connection(env, real_local_dynamics[:STATE_SIZE], synapse=IdealDelay(PREDICT_DT))
    nengo.Connection(env, real_local_dynamics[STATE_SIZE:], synapse=None)

    def calc_error(t, x):
        if t < 1:
            return [0] * DYNAMICS_SIZE 
        return x
    dynamics_error = nengo.Node(calc_error, size_in=DYNAMICS_SIZE)
    nengo.Connection(predictor.output, dynamics_error, transform=1, synapse=IdealDelay(PREDICT_DT))
    nengo.Connection(real_local_dynamics, dynamics_error, transform=-1, synapse=None)
    
    if PREDICTOR_NEURONS > 0:
        nengo.Connection(dynamics_error, predictor.dynamics_error, synapse=None)

    def write_log(t, x):
        if t > 0:
            log_writer.writerow(list(x))
    logger_node = nengo.Node(write_log, size_in=STATE_SIZE+DYNAMICS_SIZE+1)
    nengo.Connection(env, logger_node[:STATE_SIZE], synapse=None)
    nengo.Connection(dynamics_error, logger_node[STATE_SIZE:STATE_SIZE+DYNAMICS_SIZE], synapse=None)
    nengo.Connection(controller[ACTION_SIZE:], logger_node[STATE_SIZE+DYNAMICS_SIZE:], synapse=None)

# Run the simulation
carla_env.prepare(
    args.map,
    args.car,
    list(map(int, args.waypoints.split(","))),
    args.waypoints_resolution)

run_sim = nengo.Simulator(model, DT)
with run_sim:
    run_sim.run(args.simulation_time)