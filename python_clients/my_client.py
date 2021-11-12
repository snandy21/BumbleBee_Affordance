#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue
#import transforms3d

DEGREE_TO_RAD = 0.01745329252


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def convert_degrees_to_rad(carla_rotation) : 
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)

    return (roll, pitch, yaw)

def convert_xyz(carla_location) : 
    return (carla_location.x, carla_location.y, carla_location.z)

def get_relative_cordinates(ego_vehicle, actor) : 
    ego_vehicle_xyz = ego_vehicle.get_transform().location
    actor_xyz       = actor.get_transform().location
    relative_xyz    = actor_xyz - ego_vehicle_xyz
    relative_xyz    = convert_xyz(relative_xyz)

    ego_vehicle_rot     = ego_vehicle.get_transform().rotation
    (roll, pitch, yaw)  = convert_degrees_to_rad(ego_vehicle_rot)

    # print("ego xyz : ", ego_vehicle_xyz, " actor xyz : ", actor_xyz)
    # print("relative xyz : ", relative_xyz)
    # print("ego rot : ", ego_vehicle_rot, " ego rot (rad) : ", (roll, pitch, yaw))

    R = transforms3d.euler.euler2mat(roll, pitch, yaw).T
    actor_loc_relative = np.dot(R,relative_xyz)
    # print("actor loc relative : ", actor_loc_relative)

    return actor_loc_relative   
    # print("ego rot(rad) : ", ego_vehicle_rot_rad, " actor rot(rad) : ", actor_rot_rad)

def get_distance(cord): 
        return math.sqrt(
        cord[0] ** 2 + cord[1] ** 2 + cord[2] ** 2)

def get_relative_distance_for_actors(ego_vehicle, actors) : 
    actor_relative_cord = [(get_relative_cordinates(ego_vehicle, x), x) for x in actors if x.id != ego_vehicle.id]
    actor_relative_dist = [(get_distance(x[0]), x[0], x[1]) for x in actor_relative_cord]
    return actor_relative_dist
    
def measure_distance_to_vehicles(world, ego_vehicle) :
    t                       = ego_vehicle.get_transform() 
    vehicles                = world.get_actors().filter('vehicle.*')
    vehicles_relative_dist  = get_relative_distance_for_actors(ego_vehicle, vehicles)

    # for d, vehicle in sorted(vehicles):
    for d, cord, vehicle in sorted(vehicles_relative_dist): 
        if d > 20.0:            #human vision depth range = 50
            break
        # print("Close vehicle : ", d, cord, vehicle)
        print("Close vehicle : %s : d : %.3f, x : %.3f, y : %.3f, z : %.3f" %(vehicle.type_id, d, cord[0], cord[1], cord[2])) 
        # print("******* vehicle ", vehicle, " dist : ", d , " ****************")
        # print("Ego Vehicle   : ", ego_vehicle.get_transform())
        # print("Other Vehicle : ", vehicle.get_transform())
        # input()
    
def measure_distance_to_pedestrians(world, ego_vehicle) :
    t                       = ego_vehicle.get_transform() 
    walkers                 = world.get_actors().filter('walker.*')
    walkers_relative_dist   = get_relative_distance_for_actors(ego_vehicle, vehicles)

    # for d, vehicle in sorted(vehicles):
    for d, cord, walker in sorted(walkers_relative_dist): 
        if d > 20.0:            #human vision depth range = 50
            break
        print("Close walker : %s : d : %.3f, x : %.3f, y : %.3f, z : %.3f" %(walker.type_id, d, cord[0], cord[1], cord[2]) ) 
#     def distance(l): 
#         return math.sqrt(
#         (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)

#     walkers = [(distance(x.get_location()), x) for x in walkers if x.id != ego_vehicle.id]
#     for d, walker in sorted(walkers):
#         if d > 10.0:            #human vision depth range = 50
#             break
    
# def measure_vehicle_status(vehicle) : 
#     vehicle_control     = vehicle.get_control()
#     vehicle_velocity    = vehicle.get_velocity()
#     vehicle_transform   = vehicle.get_transform()
    # print("vehicle controls : ", vehicle_control)
    # print("vehicle velocity : ", vehicle_velocity)

def measure_distance_to_driving_lane(world, vehicle) :
    #waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving))
    waypoint_location = waypoint.transform.location
    waypoint_rotation = waypoint.transform.rotation
    vehicle_xyz = vehicle.get_transform().location
    relative_xyz    = waypoint_location - vehicle_xyz
    distance_of_lane   = get_distance(convert_xyz(relative_xyz))
    return distance_of_lane

def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        blueprint_library = world.get_blueprint_library()

        #####################################################
        # spawn ego vehicle
        #####################################################
        # vehicle = world.spawn_actor(
        #     # random.choice(blueprint_library.filter('vehicle.*')),
        #     random.choice(blueprint_library.filter('vehicle.bmw.isetta')),
        #     start_pose)
        bp = blueprint_library.filter('model3')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)       
        #waypoint = m.get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk)) 
        # vehicle.set_simulate_physics(False)
        # vehicle.set_simulate_physics(False)

        #####################################################
        # spawn ego vehicle RGB front facing camera
        #####################################################
        rgb_camera_blueprint = blueprint_library.find('sensor.camera.rgb')  
        # change the dimensions of the image
        rgb_camera_blueprint.set_attribute('image_size_x', '800')
        rgb_camera_blueprint.set_attribute('image_size_y', '600')
        rgb_camera_blueprint.set_attribute('fov', '110')
         # Provide the position of the sensor relative to the vehicle.
        rgb_camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        rgb_camera_sensor = world.spawn_actor(
            rgb_camera_blueprint, 
            rgb_camera_transform, 
            attach_to=vehicle)
        # add sensor to list of actors
        actor_list.append(rgb_camera_sensor)
	# ...
 
        # print("Current lane type: " + str(waypoint.lane_type))
	# Check current lane change allowed
        #print("Current Lane change:  " + str(waypoint.lane_change))
        # Left and Right lane markings
        #print("L lane marking type: " + str(waypoint.left_lane_marking.type))
        #print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
        #print("R lane marking type: " + str(waypoint.right_lane_marking.type))
        #print("R lane marking change: " + str(waypoint.right_lane_marking.lane_change))
	# ...
        #####################################################
        # spawn ego vehicle depth front facing camera
        #####################################################


        # camera_semseg = world.spawn_actor(
        #     blueprint_library.find('sensor.camera.semantic_segmentation'),
        #     carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        #     attach_to=vehicle)
        # actor_list.append(camera_semseg)

        # Create a synchronous mode context.
        # with CarlaSyncMode(world, rgb_camera_sensor, camera_semseg, fps=30) as sync_mode:
        with CarlaSyncMode(world, rgb_camera_sensor, fps=20) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                # snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Choose the next waypoint and update the car location.
                # vehicle_control = vehicle.get_control()
                # vehicle.apply_control(vehicle_control)
                # vehicle.set_autopilot(True)
                # vehicle_control = vehicle.get_control()
                # vehicle.apply_control(vehicle_control)

                #######################################
                # Measurements
                #######################################
                # measure_vehicle_status(vehicle)
                measure_distance_to_vehicles(world, vehicle)
                # measure_distance_to_pedestrians(world, vehicle)
                # print("vehicle controls : ", vehicle.get_control())
                # waypoint = random.choice(waypoint.next(1.5))
                # vehicle.set_transform(waypoint.transform)
                distance_of_the_waypoint =  measure_distance_to_driving_lane(world, vehicle)
                print("Waypoint_distance:", distance_of_the_waypoint)
                # image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                # draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
