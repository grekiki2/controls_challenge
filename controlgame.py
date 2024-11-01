#!/usr/bin/env python
import pygame
import sys
import numpy as np
from controllers import BaseController
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tinyphysics import TinyPhysicsSimulator
import os
import warnings
warnings.filterwarnings("ignore")
import random

pygame.init()
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"


W = 1400
H = 2500
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PAD = 5

class Controller(BaseController):
    def __init__(self):
        self.sim = None
        self.next_command = 0
        
    def giveSim(self, sim):
        self.sim = sim
        
    def update(self, target_lataccel, current_lataccel, state, future_plan=None):
        return self.next_command
    
def create_plot(y_data, title, xlabel, ylabel, vertical_time=False, y_lim=None, x_data=None, x_lim=None, shade_range=None):
    # Calculate figure size to match cell size
    FW = W // 2 - 2*PAD
    FH = H // 3 - 2*PAD
    if vertical_time:
        FH -= 100
    fig_width = FW / 100  
    fig_height = FH / 100
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if shade_range is not None:
        # shade_range should be a tuple of (x_values, alpha_values)
        x_vals, alphas, alphas2 = shade_range
        for i in range(len(x_vals)-1):
            ax.axvspan(x_vals[i], x_vals[i+1], alpha=alphas[i], color='green', linewidth=0)
            ax.axvspan(x_vals[i], x_vals[i+1], alpha=alphas2[i], color='red', linewidth=0)
    if x_data is None:
      x_data = range(sim.step_idx-20, sim.step_idx)
    for dataLine in y_data:
        if vertical_time:
            ax.plot(dataLine, x_data)
            if y_lim:
              ax.set_xlim(y_lim)
            if x_lim:
              ax.set_ylim(x_lim)
        else:
            ax.scatter(x_data, dataLine)
            if y_lim:
              ax.set_ylim(y_lim)
            if x_lim:
              ax.set_xlim(x_lim)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    
    plt.close(fig)
    
    surface = pygame.image.fromstring(raw_data, size, "RGB")
    return pygame.transform.scale(surface, (FW, FH))

controller = Controller()
sim = TinyPhysicsSimulator("./data/00000.csv", controller=controller, debug=False)
controller.giveSim(sim)
np.random.seed(random.randrange(2**32))
for _ in range(80):
    sim.step()
controller.next_command = sim.action_history[-1]

cell_width = W // 2
cell_height = H // 3

slider_rect = pygame.Rect(cell_width + 50, 50, cell_width - 100, 50)
slider_value = 0
drawn = False
drawn2 = False

while True:
    # screen.fill(BLACK)
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            if x > W//2 and y < H//3:
              slider_value = (x - slider_rect.left) / slider_rect.width * 2 - 1
              slider_value = np.clip(slider_value, -1, 1)
              drawn2 = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if x > W//2 and y < H//3:
                controller.next_command += 0.02 * slider_value
                sim.step()
                print(f"{sim.step_idx} cost {sim.compute_cost()['total_cost']:.2f} last {sim.compute_last_cost():.2f}")
                if sim.step_idx == 150:
                    exit()
                drawn = False
                drawn2 = False
    # Draw grid lines
    for i in range(1, 2):
        pygame.draw.line(screen, WHITE, (i * cell_width, 0), (i * cell_width, H))
    for i in range(1, 3):
        pygame.draw.line(screen, WHITE, (0, i * cell_height), (W, i * cell_height))
    # Draw slider
    pygame.draw.rect(screen, WHITE, slider_rect)  # Slider background
    pygame.draw.line(screen, BLACK, (slider_rect.left + slider_rect.width / 2, slider_rect.top), (slider_rect.left + slider_rect.width / 2, slider_rect.bottom))
    # Draw slider handle
    handle_pos = slider_rect.left + (slider_value + 1) * slider_rect.width / 2
    handle_rect = pygame.Rect(handle_pos - 5, slider_rect.top, 10, slider_rect.height)
    pygame.draw.rect(screen, GREEN, handle_rect)

    if drawn:
        if drawn2:
            continue
        meanLataccel = np.mean(sim.current_lataccel_history[-20:])
        x_vals = np.linspace(-5, 5, 1024)
        p_dist = sim.getProbDist(controller.next_command + 0.02*slider_value)
        target = sim.get_state_target_futureplan(sim.step_idx)[1]
        prev_angle = sim.current_lataccel_history[-1]

        x_vals2 = np.linspace(meanLataccel-0.5, meanLataccel+0.5, 100)
        # Vectorized cost calculations
        angleCost = 100 * (target - x_vals2)**2
        jerkCost = 100 * ((x_vals2 - prev_angle)/0.1)**2
        totalCost = 50*angleCost + jerkCost

        # Vectorized alpha calculations
        alphas = np.interp(totalCost, [0, 50], [1, 0])  # Green shading
        alphas2 = np.interp(totalCost, [100, 200], [0, 1])  # Optional: red shading for high cost

        shade_range = (x_vals2, alphas, alphas2)
        probs = create_plot([p_dist], "Probabilities", "Lateral Acceleration", "Probability", False, [0, 0.2], x_vals, [meanLataccel-0.5, meanLataccel+0.5], shade_range)
        screen.blit(probs, (cell_width+PAD, cell_height + PAD))
        pygame.display.flip()
        drawn2 = True
        continue
    
    # Get actual data from simulation
    v_ego_data = [state.v_ego for state in sim.state_history][-20:]
    a_ego_data = [state.a_ego for state in sim.state_history][-20:]

    lataccel_roll_data = [state.roll_lataccel for state in sim.state_history][-20:]
    
    # Create plots with real data
    v_ego_plot = create_plot([v_ego_data], "v_ego", "Time", "Speed", False, [0, 35])
    a_ego_plot = create_plot([a_ego_data], "a_ego", "Time", "Acceleration", False, [-2, 2])
    lataccel_roll_plot = create_plot([lataccel_roll_data], "lataccel_roll", "Time", "Lateral Acceleration", False, [-0.5, 0.5])
    
    # Bottom right plot showing target vs current lataccel
    meanLataccel = np.mean(sim.current_lataccel_history[-20:])
    target_vs_current = create_plot(
        [sim.current_lataccel_history[-20:], sim.target_lataccel_history[-20:]], 
        "Lataccels", 
        "Acceleration",
        "Time",
        True,
        [meanLataccel-0.5, meanLataccel+0.5]
    )
    cmds = create_plot([sim.action_history[-20:]], "Commands", "Command", "Time", True, [-1, 1])
    # Display plots
    screen.blit(v_ego_plot, (PAD, PAD))
    screen.blit(a_ego_plot, (PAD, cell_height+PAD))
    screen.blit(lataccel_roll_plot, (PAD, 2 * cell_height + PAD))

    screen.blit(cmds, (cell_width+PAD, 100 + PAD))
    screen.blit(target_vs_current, (cell_width+PAD, 2 * cell_height + PAD))
    drawn = True
    
    pygame.display.flip()
