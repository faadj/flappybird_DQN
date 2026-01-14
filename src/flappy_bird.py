
from itertools import cycle
from numpy.random import randint
from pygame import Rect, init, time, display, font, draw
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
from pygame.event import pump
import numpy as np
import math


class FlappyBird(object):
    init()
    fps_clock = time.Clock()
    screen_width = 288
    screen_height = 512
    screen = display.set_mode((screen_width, screen_height))
    display.set_caption('Deep Q-Network Flappy Bird')
    base_image = load('assets/sprites/base.png').convert_alpha()
    background_image = load('assets/sprites/background-black.png').convert()

    pipe_images = [rotate(load('assets/sprites/pipe-green.png').convert_alpha(), 180),
                   load('assets/sprites/pipe-green.png').convert_alpha()]
    bird_images = [load('assets/sprites/redbird-upflap.png').convert_alpha(),
                   load('assets/sprites/redbird-midflap.png').convert_alpha(),
                   load('assets/sprites/redbird-downflap.png').convert_alpha()]

    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_images]
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]

    # --- CONFIGURATION ---
    fps = 300
    pipe_velocity_x = -4
    min_velocity_y = -8
    max_velocity_y = 10
    downward_speed = 1
    upward_speed = -9

    # LIDAR CONFIGURATION
    lidar_count = 16
    lidar_scope = 120
    lidar_max_dist = 300

    bird_index_generator = cycle([0, 1, 2, 1])

    def __init__(self):
        self.font = font.SysFont(None, 20, bold=True)
        # New Stats
        self.easy_mode_best = 0
        self.hard_mode_best = 0

        # Training Progress (Set by train.py)
        self.training_progress = 0.0

        # --- CURRICULUM VARIABLES ---
        self.pipe_gap_size = 200
        self.vertical_variance = 0
        # ----------------------------

        self.reset()

    def reset(self):
        self.iter = self.bird_index = self.score = 0
        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()
        self.pipe_width = self.pipe_images[0].get_width()
        self.pipe_height = self.pipe_images[0].get_height()

        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)

        self.base_x = 0
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()

        pipes = [self.generate_pipe(), self.generate_pipe()]
        pipes[0]["x_upper"] = pipes[0]["x_lower"] = self.screen_width
        pipes[1]["x_upper"] = pipes[1]["x_lower"] = self.screen_width * 1.5
        self.pipes = pipes

        self.current_velocity_y = 0
        self.is_flapped = False

    def generate_pipe(self):
        x = self.screen_width + 10
        min_y = int(self.base_y * 0.2)
        max_y = int(self.base_y * 0.8 - self.pipe_gap_size)
        center_y = int((self.base_y - self.pipe_gap_size) / 2)

        random_y = randint(min_y, max_y)
        gap_y = int(center_y + (random_y - center_y) * self.vertical_variance)

        return {"x_upper": x, "y_upper": gap_y - self.pipe_height, "x_lower": x, "y_lower": gap_y + self.pipe_gap_size}

    def is_collided(self):
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
        pipe_boxes = []
        for pipe in self.pipes:
            pipe_boxes.append(Rect(pipe["x_upper"], pipe["y_upper"], self.pipe_width, self.pipe_height))
            pipe_boxes.append(Rect(pipe["x_lower"], pipe["y_lower"], self.pipe_width, self.pipe_height))
            if bird_bbox.collidelist(pipe_boxes) == -1:
                return False
            for i in range(2):
                cropped_bbox = bird_bbox.clip(pipe_boxes[i])
                min_x1 = cropped_bbox.x - bird_bbox.x
                min_y1 = cropped_bbox.y - bird_bbox.y
                min_x2 = cropped_bbox.x - pipe_boxes[i].x
                min_y2 = cropped_bbox.y - pipe_boxes[i].y
                if np.any(self.bird_hitmask[self.bird_index][min_x1:min_x1 + cropped_bbox.width,
                min_y1:min_y1 + cropped_bbox.height] * self.pipe_hitmask[i][min_x2:min_x2 + cropped_bbox.width,
                min_y2:min_y2 + cropped_bbox.height]):
                    return True
        return False

    def cast_ray(self, start_x, start_y, angle_rad, obstacles):
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        for dist in range(0, self.lidar_max_dist, 10):
            check_x = start_x + dx * dist
            check_y = start_y + dy * dist
            if check_y < 0 or check_y > self.base_y:
                return dist / self.lidar_max_dist
            for obs in obstacles:
                if obs.collidepoint(check_x, check_y):
                    return dist / self.lidar_max_dist
        return 1.0

    def get_lidar_state(self):
        center_x = self.bird_x + self.bird_width / 2
        center_y = self.bird_y + self.bird_height / 2
        obstacles = []
        for pipe in self.pipes:
            obstacles.append(Rect(pipe["x_upper"], pipe["y_upper"], self.pipe_width, self.pipe_height))
            obstacles.append(Rect(pipe["x_lower"], pipe["y_lower"], self.pipe_width, self.pipe_height))

        readings = []
        start_angle = -self.lidar_scope / 2
        step = self.lidar_scope / (self.lidar_count - 1)
        for i in range(self.lidar_count):
            deg = start_angle + (step * i)
            rad = math.radians(deg)
            readings.append(self.cast_ray(center_x, center_y, rad, obstacles))
        readings.append(self.current_velocity_y / 10.0)
        return np.array(readings, dtype=np.float32)

    def next_frame(self, action):
        pump()
        reward = 0.1
        terminal = False

        if action == 1:
            self.current_velocity_y = self.upward_speed
            self.is_flapped = True
            reward -= 0.05

        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + self.pipe_width / 2
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1

                # --- REWARD LOGIC (INFINITE PLAY) ---
                reward += 1  # Base reward for passing

                # Streak Bonus
                if self.score % 5 == 0:
                    reward += 1

                    # Milestone Bonus (Achievement Points)
                #  at 100/500, we give huge points
                if self.score % 100 == 0:
                    reward += 10

                    # --- NO RESET HERE: Game continues forever ---

                # --- BEST SCORE LOGIC ---
                if self.pipe_gap_size > 150:  # Easy Mode
                    if self.score > self.easy_mode_best: self.easy_mode_best = self.score
                else:  # Hard/Medium Mode
                    if self.score > self.hard_mode_best: self.hard_mode_best = self.score
                break

        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        if self.is_flapped:
            self.is_flapped = False
        self.bird_y += min(self.current_velocity_y, self.bird_y - self.current_velocity_y - self.bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_velocity_x
            pipe["x_lower"] += self.pipe_velocity_x
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.generate_pipe())
        if self.pipes[0]["x_lower"] < -self.pipe_width:
            del self.pipes[0]

        if self.is_collided():
            terminal = True
            reward = -10
            self.reset()

        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(self.bird_images[self.bird_index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            self.screen.blit(self.pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            self.screen.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))

        # --- DISPLAY STATS ---
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        easy_best_text = self.font.render(f"Easy Best: {self.easy_mode_best}", True, (0, 255, 0))
        hard_best_text = self.font.render(f"Hard Best: {self.hard_mode_best}", True, (255, 100, 100))
        reward_text = self.font.render(f"Reward: {reward:.2f}", True, (255, 255, 255))

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(easy_best_text, (10, 30))
        self.screen.blit(hard_best_text, (10, 50))
        self.screen.blit(reward_text, (10, 70))

        # --- PROGRESS BAR ---
        bar_width = 200
        bar_height = 10
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = self.screen_height - 30

        draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        fill_width = int(bar_width * self.training_progress)
        draw.rect(self.screen, (0, 150, 255), (bar_x, bar_y, fill_width, bar_height))
        draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2)

        pct_text = self.font.render(f"{int(self.training_progress * 100)}%", True, (255, 255, 255))
        self.screen.blit(pct_text, (bar_x + bar_width + 10, bar_y - 2))

        image = array3d(display.get_surface())
        display.update()
        self.fps_clock.tick(self.fps)

        state_vector = self.get_lidar_state()
        return image, reward, terminal, state_vector