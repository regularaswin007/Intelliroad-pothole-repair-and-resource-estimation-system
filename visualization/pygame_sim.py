"""
Pygame Real-Time Simulation
Watch the repair agent move through the network!
"""

import pygame
import numpy as np
import sys
sys.path.append('..')

from environment.moving_agent_env import MovingRepairAgentEnv
from models.drl_agent import PotholeRepairAgent

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
GRAY = (150, 150, 150)

class PygameSimulation:
    """Real-time visualization of moving repair agent"""
    
    def __init__(self, env, agent=None):
        pygame.init()
        self.width = 1400
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🚗 Moving Repair Agent - DRL Simulation")
        
        self.env = env
        self.agent = agent
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # Calculate segment positions (circular layout)
        self.segment_positions = self._calculate_positions()
        
        # Animation state
        self.agent_display_pos = None
        self.moving = False
        self.move_progress = 0
        self.target_pos = None
        
    def _calculate_positions(self):
        """Position segments in a circle"""
        positions = {}
        center_x, center_y = 500, 400
        radius = 300
        
        for i in range(self.env.num_segments):
            angle = (2 * np.pi * i / self.env.num_segments) - np.pi/2
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions[i] = (int(x), int(y))
        
        return positions
    
    def draw_road_network(self):
        """Draw road segments and connections"""
        # Draw edges
        for i in range(self.env.num_segments):
            for j in range(i + 1, self.env.num_segments):
                if self.env.road_network.graph.has_edge(i, j):
                    pygame.draw.line(
                        self.screen,
                        GRAY,
                        self.segment_positions[i],
                        self.segment_positions[j],
                        2
                    )
        
        # Draw segments
        for seg_id, pos in self.segment_positions.items():
            segment = self.env.road_network.get_segment(seg_id)
            
            # Color based on condition
            if segment.condition_index >= 75:
                color = GREEN
            elif segment.condition_index >= 50:
                color = YELLOW
            elif segment.condition_index >= 25:
                color = ORANGE
            else:
                color = RED
            
            pygame.draw.circle(self.screen, color, pos, 15)
            pygame.draw.circle(self.screen, BLACK, pos, 15, 2)
            
            # Segment ID
            text = self.font.render(str(seg_id), True, BLACK)
            self.screen.blit(text, (pos[0] - 8, pos[1] - 8))
    
    def draw_potholes(self):
        """Draw potholes on segments"""
        for pothole in self.env.potholes:
            pos = self.segment_positions[pothole.segment_id]
            
            # Pothole color by severity
            severity_colors = {
                'minor': YELLOW,
                'moderate': ORANGE,
                'severe': (255, 100, 0),
                'critical': RED
            }
            color = severity_colors.get(pothole.severity, RED)
            
            # Pulsing effect
            pulse = int(5 + 3 * np.sin(pygame.time.get_ticks() / 200))
            pygame.draw.circle(self.screen, color, pos, pulse, 0)
    
    def draw_agent(self):
        """Draw repair agent (truck)"""
        if self.agent_display_pos is None:
            self.agent_display_pos = self.segment_positions[self.env.agent_position]
        
        # Interpolate position if moving
        if self.moving and self.target_pos:
            current = np.array(self.agent_display_pos)
            target = np.array(self.target_pos)
            self.agent_display_pos = tuple(
                (current + (target - current) * self.move_progress).astype(int)
            )
            self.move_progress += 0.05
            
            if self.move_progress >= 1.0:
                self.moving = False
                self.move_progress = 0
                self.agent_display_pos = self.target_pos
        
        # Draw truck
        truck_size = 20
        pygame.draw.rect(
            self.screen,
            BLUE,
            (self.agent_display_pos[0] - truck_size//2,
             self.agent_display_pos[1] - truck_size//2,
             truck_size, truck_size)
        )
        
        # Draw "🚗" emoji-style
        text = self.font.render("🚗", True, WHITE)
        self.screen.blit(text, 
                        (self.agent_display_pos[0] - 10,
                         self.agent_display_pos[1] - 10))
    
    def draw_dashboard(self, info):
        """Draw stats dashboard"""
        dashboard_x = 1050
        dashboard_y = 50
        
        # Title
        title = self.title_font.render("Dashboard", True, BLACK)
        self.screen.blit(title, (dashboard_x, dashboard_y))
        
        # Stats
        stats = [
            f"Step: {self.env.current_step}/{self.env.episode_length}",
            f"Position: Segment {self.env.agent_position}",
            f"Fuel: {self.env.agent_fuel:.1f}/1000",
            f"Repair Capacity: {self.env.repair_capacity}/20",
            f"",
            f"Total Repaired: {self.env.total_repaired}",
            f"Active Potholes: {len(self.env.potholes)}",
            f"Critical: {info.get('critical_potholes', 0)}",
            f"",
            f"Distance: {self.env.total_distance:.1f}m",
        ]
        
        y_offset = 100
        for stat in stats:
            text = self.font.render(stat, True, BLACK)
            self.screen.blit(text, (dashboard_x, dashboard_y + y_offset))
            y_offset += 30
    
    def run(self, num_steps=1000, fps=10):
        """Run simulation"""
        obs, _ = self.env.reset()
        running = True
        step = 0
        paused = False
        
        while running and step < num_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            if not paused:
                # Get action from agent or random
                if self.agent:
                    action, _ = self.agent.model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                # Take step
                old_position = self.env.agent_position
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Animate movement
                if info.get('moved') and old_position != self.env.agent_position:
                    self.target_pos = self.segment_positions[self.env.agent_position]
                    self.moving = True
                    self.move_progress = 0
                
                step += 1
                
                if terminated or truncated:
                    print(f"\nSimulation ended at step {step}")
                    print(f"Total repaired: {self.env.total_repaired}")
                    break
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_road_network()
            self.draw_potholes()
            self.draw_agent()
            self.draw_dashboard(info)
            
            # Show pause message
            if paused:
                pause_text = self.title_font.render("PAUSED (SPACE to resume)", True, RED)
                self.screen.blit(pause_text, (400, 50))
            
            pygame.display.flip()
            self.clock.tick(fps)
        
        pygame.quit()


# Main execution
if __name__ == '__main__':
    print("Initializing Moving Repair Agent Simulation...")
    
    # Create environment
    env = MovingRepairAgentEnv(num_segments=30, episode_length=500)
    
    # Try to load trained agent
    try:
        agent = PotholeRepairAgent(env=env, verbose=0)
        agent.load('../models/best_model')
        print("✓ Loaded trained agent")
    except:
        print("! No trained agent, using random policy")
        agent = None
    
    # Run simulation
    sim = PygameSimulation(env, agent)
    print("\nStarting simulation...")
    print("Controls: SPACE = pause/resume, ESC = quit")
    sim.run(num_steps=500, fps=10)
