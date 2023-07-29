from enum import IntEnum
import math
import numpy as np
import networkx as nx
from collections import deque

import carla
import nengo

SIMULATOR_DT = 1 / 120
STATE_SIZE = 9
ACTION_SIZE = 2


def bicycle_model_dynamics_center(x, vehicle_wheelbase=2.3, max_steer=70):
    """
    Calculate the dynamics of a vehicle using the bicycle model with center of gravity.

    Args:
        x (list): State vector [throttle, steer, yaw_cos, yaw_sin, vx, vy, ax, ay, vr].
        vehicle_wheelbase (float, optional): Vehicle wheelbase in meters. Defaults to 2.3.
        max_steer (int, optional): Maximum steering angle in degrees. Defaults to 70.

    Returns:
        tuple: Tuple containing the derivatives of the state variables (dx, dy, dyaw, dvx, dvy, dax, day, dvr).
    """

    throttle, steer, yaw_cos, yaw_sin, vx, vy, _, _, _ = x

    v = math.hypot(vx, vy)
    yaw = math.atan2(yaw_sin, yaw_cos)
    steer_angle = steer * math.radians(max_steer)
    rear_wheel_to_cg = vehicle_wheelbase / 2

    steer_angle_tan = math.tan(steer_angle)

    beta = math.atan2(rear_wheel_to_cg * steer_angle_tan, vehicle_wheelbase)

    # These parameters are not used in this model
    dax = 0
    day = 0
    dvr = 0

    dvx = throttle * yaw_cos
    dvy = throttle * yaw_sin
    dx = v * math.cos(yaw + beta)
    dy = v * math.sin(yaw + beta)
    dyaw = v * steer_angle_tan * math.cos(beta) / vehicle_wheelbase

    return dx, dy, dyaw, dvx, dvy, dax, day, dvr


def bicycle_model_state_center(x, u, dt, vehicle_wheelbase=2.3, max_steer=70):
    """
    Update the state of a vehicle using the bicycle model with center of gravity.

    Args:
        x (list): State vector [px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr].
        u (list): Control input vector [throttle, steer].
        dt (float): Time step for integration.
        vehicle_wheelbase (float, optional): Vehicle wheelbase in meters. Defaults to 2.3.
        max_steer (int, optional): Maximum steering angle in degrees. Defaults to 70.

    Returns:
        list: Updated state vector [px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr].
    """

    px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr = x
    throttle, steer = u

    dx, dy, dyaw, dvx, dvy, dax, day, dvr = bicycle_model_dynamics_center(
        [throttle, steer, yaw_cos, yaw_sin, vx, vy, ax, ay, vr],
        vehicle_wheelbase,
        max_steer)

    yaw = math.atan2(yaw_sin, yaw_cos)

    px += dx * dt
    py += dy * dt
    yaw += dyaw * dt
    vx += dvx * dt
    vy += dvy * dt
    ax += dax * dt
    ay += day * dt
    vr += dvr * dt

    yaw_cos = math.cos(yaw)
    yaw_sin = math.sin(yaw)

    return px, py, yaw_cos, yaw_sin, vx, vy, ax, ay, vr


class CarlaInterface(nengo.Process):
    """
    The CarlaInterface class provides an interface to interact with the CARLA simulator.
    """
    def __init__(
            self,
            default_dt=SIMULATOR_DT,
            seed=None,
            host='localhost',
            port=2000):
        """
        Initialize the CarlaInterface.

        Args:
            default_dt (float): Default simulation time step.
            seed (int): Seed for random number generation.
            host (str): Host address of the CARLA server.
            port (int): Port number of the CARLA server.
        """

        super().__init__(ACTION_SIZE, STATE_SIZE, default_dt, seed)

        self.__dt = default_dt
        self.__client = carla.Client(host=host, port=port)
        self.__client.set_timeout(60.0)
        self.__route = None
        self.__camera = None
        self.vehicle = None
        self.is_ready = False

    def prepare(self, world, vehicle, waypoints, waypoints_resolution=1):
        """
        Prepare the simulation by loading the world, initializing it, creating a route, spawning a vehicle, and drawing the route.

        Args:
            world (str): Name of the world to load in CARLA.
            vehicle (str): Name of the vehicle to spawn.
            waypoints (list): List of waypoint indices for the route.
            waypoints_resolution (int): Resolution of the waypoints in meters.
        """

        self.__world = self.__client.load_world(world)
        self.__initialize_world()
        self.__route = self.__create_route(waypoints, waypoints_resolution)
        self.__spawn_vehicle(vehicle, waypoints[0])
        self.__draw_route(self.__route)
        self.is_ready = True
        self.__waypoints_resolution = waypoints_resolution

    def disconnect(self):
        """
        Disconnect from the CARLA server and reload the world.
        """

        if self.__client is not None:
            self.__client.reload_world(True)

    def get_waypoints(self, points):
        """
        Get the specified number of waypoints from the current location.

        Args:
            points (int): Number of waypoints to return.

        Returns:
            list: List of waypoints.
        """
        
        if self.__route is None:
            return None
        return list(self.__route)[:points]

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """
        Create a step function for the Nengo process.

        Args:
            shape_in (tuple): Shape of the input array.
            shape_out (tuple): Shape of the output array.
            dt (float): Time step for the simulation.
            rng (np.random.RandomState): Random number generator.
            state (ndarray): State array for the Nengo process.

        Returns:
            function: Step function for the Nengo process.
        """

        def step(t, x):
            return self.sim_step(x)

        return step

    def sim_step(self, u):
        """
        Perform a simulation step in CARLA.

        Args:
            u (ndarray): Control input array where the first element is the throttle and the second element is the steering angle ranging from -1 to 1.

        Returns:
            list: List of vehicle state values.
        """

        u = np.clip(u, -1, 1)

        if u[0] < 0:
            control = carla.VehicleControl(throttle=0, brake=abs(float(u[0])), steer=float(u[1]))
        else:
            control = carla.VehicleControl(throttle=float(u[0]), brake=0, steer=float(u[1]))

        self.vehicle.apply_control(control)

        self.__world.tick()

        self.__follow_vehicle()

        passed_waypoints = self.__update_waypoints()
        self.__route.rotate(-passed_waypoints)

        state =  self.get_state()

        return state
    
    def get_state(self):
        """
        Get the current state of the vehicle.

        Returns:
            list: List of vehicle state values.
        """

        return [
            self.vehicle.get_location().x,
            self.vehicle.get_location().y,
            math.cos(math.radians(self.vehicle.get_transform().rotation.yaw)),
            math.sin(math.radians(self.vehicle.get_transform().rotation.yaw)),
            self.vehicle.get_velocity().x,
            self.vehicle.get_velocity().y,
            self.vehicle.get_acceleration().x,
            self.vehicle.get_acceleration().y,
            self.vehicle.get_angular_velocity().z,
        ]

    def get_road_coeff(self, look_distance=40, degree=3):
        """
        Get the coefficients of the polynomial fitting the road ahead of the vehicle.

        Args:
            look_distance (int): Look-ahead distance in meters.
            degree (int): Degree of the polynomial to fit.

        Returns:
            ndarray: Coefficients of the fitted polynomial.
        """

        points = look_distance // self.__waypoints_resolution
        road = self.__get_aligned_road()[:points]

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = np.deg2rad(-vehicle_transform.rotation.yaw)

        R = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw)],
                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw)]])

        positions = np.array([[p.transform.location.x - vehicle_location.x, p.transform.location.y - vehicle_location.y] for p in road])
        rotated_positions = np.matmul(R, positions.T).T
        road_locations = [carla.Location(x=p[0], y=p[1], z=0) for p in rotated_positions]

        return np.polyfit([p.x for p in road_locations],
                          [p.y for p in road_locations],
                          degree)
    
    def get_road(self, look_distance=40):
        """
        Get the road points ahead of the vehicle.

        Args:
            look_distance (int): Look-ahead distance in meters.

        Returns:
            ndarray: Array of road points.
        """

        points = look_distance // self.__waypoints_resolution
        road = self.__get_aligned_road()[:points]
        return np.array([[p.transform.location.x, p.transform.location.y] for p in road])

    def plot_point(self, location: carla.Location, life_time=None, size=0.1, color=carla.Color(255, 0, 0)):
        """
        Plot a point in the CARLA world.

        Args:
            location (carla.Location): Location of the point.
            life_time (float): Life time of the point in seconds.
            size (float): Size of the point.
            color (carla.Color): Color of the point.
        """

        self.__world.debug.draw_point(
            location,
            color=color,
            size=size,
            life_time=2 * self.__dt if life_time is None else life_time,
            persistent_lines=True)

    def __get_aligned_road(self):
        """
        Get the road aligned with the vehicle's position.

        Returns:
            list: List of aligned waypoints.
        """

        new_waypoints = self.__route.copy()
        new_waypoints.reverse()
        new_waypoints = deque(new_waypoints)

        veh_location = self.vehicle.get_location()

        i = 0
        min_waypoint_index = 0
        min_dist = 999
        last_dist = 999
        for waypoint in new_waypoints:
            dist = veh_location.distance(waypoint.transform.location)

            i += 1

            if dist > last_dist:
                break
            elif dist < min_dist:
                min_dist = dist
                min_waypoint_index = i
            last_dist = dist

        new_waypoints.rotate(-min_waypoint_index)
        lst = list(new_waypoints)
        lst.reverse()
        return lst

    def __initialize_world(self):
        """
        Initialize the CARLA world with fixed time step and synchronous mode. For more info see: https://carla.readthedocs.io/en/0.9.14/adv_synchrony_timestep/#physics-determinism
        """
                
        settings = self.__world.get_settings()
        settings.fixed_delta_seconds = self.__dt
        settings.synchronous_mode = True
        self.__world.apply_settings(settings)
        self.__client.reload_world(False)  # reload map keeping the world settings
        self.__map = self.__world.get_map()

    def __create_route(self, waypoints, waypoints_resolution):
        """
        Create a route using the given waypoints and resolution.

        Args:
            waypoints (list): List of spawn points indices for the route.
            waypoints_resolution (int): Resolution of the waypoints in meters.

        Returns:
            deque: Deque containing the generated route.
        """
                
        spawn_points = self.__map.get_spawn_points()

        planner = GlobalRoutePlanner(self.__map, waypoints_resolution)
        route = []

        for i in range(len(waypoints) - 1):
            start = spawn_points[waypoints[i]].location
            end = spawn_points[waypoints[i + 1]].location
            route += list(map(lambda x: x[0], planner.trace_route(start, end)))

        return deque(route)

    def __spawn_vehicle(self, vehicle, point):
        """
        Spawn a vehicle at the specified spawn point.

        Args:
            vehicle (str): Name of the vehicle to spawn.
            point (int): Index of the spawn point.
        """
                
        blueprint_library = self.__world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle)[0]
        spawn_point = self.__map.get_spawn_points()[point]
        self.vehicle = self.__world.spawn_actor(vehicle_bp, spawn_point)

    def __draw_route(self, points):
        """
        Draw the route in the CARLA world using debug lines.

        Args:
            points (list): List of waypoints in the route.
        """

        for i in range(len(points) - 1):

            loc = points[i].transform.location
            loc = carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.5)

            loc2 = points[i + 1].transform.location
            loc2 = carla.Location(x=loc2.x, y=loc2.y, z=loc2.z + 0.5)

            self.__world.debug.draw_line(
                loc,
                loc2,
                color=carla.Color(r=0, g=0, b=255),
                life_time=0,
                thickness=0.1,
                persistent_lines=True)

    def __update_waypoints(self):
        """
        Update the waypoints based on the vehicle's position and speed.

        Returns:
            int: Number of waypoints removed from the route.
        """
                
        veh_location = self.vehicle.get_location()
        veh_velocity = self.vehicle.get_velocity()
        vehicle_speed = math.sqrt(veh_velocity.x ** 2 + veh_velocity.y ** 2 + veh_velocity.z ** 2)
        min_distance = 3 + 0.5 * vehicle_speed

        num_waypoint_removed = 0
        for waypoint in self.__route:
            if len(self.__route) - num_waypoint_removed == 1:
                min_distance_threshold = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance_threshold = min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance_threshold:
                num_waypoint_removed += 1
            else:
                break

        return num_waypoint_removed

    def __follow_vehicle(self):
        """
        Set the spectator view to follow the vehicle from above.
        """

        if self.__camera is None:
            camera_bp = self.__world.get_blueprint_library().find('sensor.other.collision')
            camera_transform = carla.Transform(carla.Location(x=3, z=30), carla.Rotation(pitch=-90))
            self.__camera = self.__world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.__world.get_spectator().set_transform(self.__camera.get_transform())

    def __del__(self):
        """
        Destructor for the CarlaInterface class. Disconnects from the server.
        """

        self.disconnect()

# The following classes took from CARLA PythonAPI examples

class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class GlobalRoutePlanner(object):
    """
    This class provides a very high level route plan.
    """

    def __init__(self, wmap, sampling_resolution):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None

        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

        # Build the graph
        self._build_topology()
        self._build_graph()
        self._find_loose_ends()
        self._lane_change_link()

    def trace_route(self, origin, destination):
        """
        This method returns list of (carla.Waypoint, RoadOption)
        from origin to destination
        """
        route_trace = []
        route = self._path_search(origin, destination)
        current_waypoint = self._wmap.get_waypoint(origin)
        destination_waypoint = self._wmap.get_waypoint(destination)

        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            edge = self._graph.edges[route[i], route[i+1]]
            path = []

            if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
                route_trace.append((current_waypoint, road_option))
                exit_wp = edge['exit_waypoint']
                n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self._graph.edges[n1, n2]
                if next_edge['path']:
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path'])-1, closest_index+5)
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                route_trace.append((current_waypoint, road_option))

            else:
                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                closest_index = self._find_closest_in_list(current_waypoint, path)
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*self._sampling_resolution:
                        break
                    elif len(route)-i <= 2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:
                        destination_index = self._find_closest_in_list(destination_waypoint, path)
                        if closest_index > destination_index:
                            break

        return route_trace

    def _build_topology(self):
        """
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects with the following attributes

        - entry (carla.Waypoint): waypoint of entry point of road segment
        - entryxyz (tuple): (x,y,z) of entry point of road segment
        - exit (carla.Waypoint): waypoint of exit point of road segment
        - exitxyz (tuple): (x,y,z) of exit point of road segment
        - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        """
        self._topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    next_ws = w.next(self._sampling_resolution)
                    if len(next_ws) == 0:
                        break
                    w = next_ws[0]
            else:
                next_wps = wp1.next(self._sampling_resolution)
                if len(next_wps) == 0:
                    continue
                seg_dict['path'].append(next_wps[0])
            self._topology.append(seg_dict)

    def _build_graph(self):
        """
        This function builds a networkx graph representation of topology, creating several class attributes:
        - graph (networkx.DiGraph): networkx graph representing the world map, with:
            Node properties:
                vertex: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """

        self._graph = nx.DiGraph()
        self._id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self._road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            # Adding edge with attributes
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=self.__vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW)

    def _find_loose_ends(self):
        """
        This method finds road segments that have an unconnected end, and
        adds them to the internal graph representation
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        for segment in self._topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
            if road_id in self._road_id_to_edge \
                    and section_id in self._road_id_to_edge[road_id] \
                    and lane_id in self._road_id_to_edge[road_id][section_id]:
                pass
            else:
                count_loose_ends += 1
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                n1 = self._id_map[exit_xyz]
                n2 = -1*count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = end_wp.next(hop_resolution)
                path = []
                while next_wp is not None and next_wp \
                        and next_wp[0].road_id == road_id \
                        and next_wp[0].section_id == section_id \
                        and next_wp[0].lane_id == lane_id:
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=end_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=end_wp.is_junction, type=RoadOption.LANEFOLLOW)

    def _lane_change_link(self):
        """
        This method places zero cost links in the topology graph
        representing availability of lane changes.
        """

        for segment in self._topology:
            left_found, right_found = False, False

            for waypoint in segment['path']:
                if not segment['entry'].is_junction:
                    next_waypoint, next_road_option, next_segment = None, None, None

                    if waypoint.right_lane_marking and waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:
                        next_waypoint = waypoint.get_right_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANERIGHT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                right_found = True
                    if waypoint.left_lane_marking and waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                        next_waypoint = waypoint.get_left_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                left_found = True
                if left_found and right_found:
                    break

    def _localize(self, location):
        """
        This function finds the road segment that a given location
        is part of, returning the edge it belongs to
        """
        waypoint = self._wmap.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass
        return edge

    def _distance_heuristic(self, n1, n2):
        """
        Distance heuristic calculator for path searching
        in self._graph
        """
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1-l2)

    def _path_search(self, origin, destination):
        """
        This function finds the shortest path connecting origin and destination
        using A* search with distance heuristic.
        origin      :   carla.Location object of start position
        destination :   carla.Location object of of end position
        return      :   path as list of node ids (as int) of the graph self._graph
        connecting origin and destination
        """
        start, end = self._localize(origin), self._localize(destination)

        route = nx.astar_path(
            self._graph, source=start[0], target=end[0],
            heuristic=self._distance_heuristic, weight='length')
        route.append(end[1])
        return route

    def _successive_last_intersection_edge(self, index, route):
        """
        This method returns the last successive intersection edge
        from a starting index on the route.
        This helps moving past tiny intersection edges to calculate
        proper turn decisions.
        """

        last_intersection_edge = None
        last_node = None
        for node1, node2 in [(route[i], route[i+1]) for i in range(index, len(route)-1)]:
            candidate_edge = self._graph.edges[node1, node2]
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break

        return last_node, last_intersection_edge

    def _turn_decision(self, index, route, threshold=math.radians(35)):
        """
        This method returns the turn decision (RoadOption) for pair of edges
        around current index of route list
        """

        decision = None
        previous_node = route[index-1]
        current_node = route[index]
        next_node = route[index+1]
        next_edge = self._graph.edges[current_node, next_node]
        if index > 0:
            if self._previous_decision != RoadOption.VOID \
                    and self._intersection_end_node > 0 \
                    and self._intersection_end_node != previous_node \
                    and next_edge['type'] == RoadOption.LANEFOLLOW \
                    and next_edge['intersection']:
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._graph.edges[previous_node, current_node]
                calculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge[
                    'intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']
                if calculate_turn:
                    last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                    self._intersection_end_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge
                    cv, nv = current_edge['exit_vector'], next_edge['exit_vector']
                    if cv is None or nv is None:
                        return next_edge['type']
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        select_edge = self._graph.edges[current_node, neighbor]
                        if select_edge['type'] == RoadOption.LANEFOLLOW:
                            if neighbor != route[index+1]:
                                sv = select_edge['net_vector']
                                cross_list.append(np.cross(cv, sv)[2])
                    next_cross = np.cross(cv, nv)[2]
                    deviation = math.acos(np.clip(
                        np.dot(cv, nv)/(np.linalg.norm(cv)*np.linalg.norm(nv)), -1.0, 1.0))
                    if not cross_list:
                        cross_list.append(0)
                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge['type']

        else:
            decision = next_edge['type']

        self._previous_decision = decision
        return decision

    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        min_distance = float('inf')
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    def __vector(self, location_1, location_2):
        """
        Returns the unit vector from location_1 to location_2

            :param location_1, location_2: carla.Location objects
        """
        x = location_2.x - location_1.x
        y = location_2.y - location_1.y
        z = location_2.z - location_1.z
        norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

        return [x / norm, y / norm, z / norm]
