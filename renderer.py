import pygame
import numpy as np
from typing import Dict, Optional, Tuple



from world import World, EMPTY, WALL, SHELF, DROP

class WarehouseRenderer:
    """
    Pygame-based renderer for warehouse environment.

    Features:
    - Real-time visualization
    - Color-coded robots (carrying vs empty)
    - FPS display
    - Metrics overlay
    - Screenshot capability
    """

   
    COLORS = {
        'background': (240, 240, 240),  # Light gray
        'grid': (200, 200, 200),  # Grid lines
        'wall': (60, 60, 60),  # Dark gray
        'shelf': (139, 69, 19),  # Brown
        'drop': (34, 139, 34),  # Green
        'robot_empty': (70, 130, 180),  # Steel blue
        'robot_carrying': (255, 140, 0),  # Dark orange
        'text': (0, 0, 0),  # Black
        'text_bg': (255, 255, 255, 220),  # Semi-transparent white
    }

    
    ROBOT_COLORS = [
        (70, 130, 180),  # Steel blue
        (220, 20, 60),  # Crimson
        (255, 215, 0),  # Gold
        (147, 112, 219),  # Medium purple
        (0, 191, 255),  # Deep sky blue
        (255, 105, 180),  # Hot pink
        (50, 205, 50),  # Lime green
        (255, 69, 0),  # Orange red
        (138, 43, 226),  # Blue violet
        (0, 206, 209),  # Dark turquoise
    ]

    def __init__(
            self,
            world: 'World',
            cell_size: int = 50,
            fps: int = 10,
            show_grid: bool = True,
            show_metrics: bool = True
    ):
        """
        Initialize pygame renderer.

        Args:
            world: World instance to render
            cell_size: Size of each grid cell in pixels
            fps: Target frames per second
            show_grid: Whether to show grid lines
            show_metrics: Whether to show metrics overlay
        """
        pygame.init()
        pygame.font.init()

        self.world = world
        self.cell_size = cell_size
        self.fps = fps
        self.show_grid = show_grid
        self.show_metrics = show_metrics

        
        self.width = world.width * cell_size
        self.height = world.height * cell_size
        self.metrics_height = 120 if show_metrics else 0

        
        self.screen = pygame.display.set_mode(
            (self.width, self.height + self.metrics_height)
        )
        pygame.display.set_caption("Multi-Agent Warehouse")

        
        self.font = pygame.font.SysFont('Arial', 16)
        self.font_small = pygame.font.SysFont('Arial', 12)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)

        
        self.clock = pygame.time.Clock()

        
        self.robot_colors = {}
        for i, robot_id in enumerate(world.robots.keys()):
            self.robot_colors[robot_id] = self.ROBOT_COLORS[i % len(self.ROBOT_COLORS)]

    def render(self, pause_ms: Optional[int] = None) -> bool:
        """
        Render current state of world.

        Args:
            pause_ms: If provided, pause for this many milliseconds

        Returns:
            False if user closed window, True otherwise
        """
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_s:
                    self.save_screenshot()

        
        self.screen.fill(self.COLORS['background'])

      
        self._draw_grid()

        
        self._draw_static_entities()

        
        self._draw_robots()

        
        if self.show_metrics:
            self._draw_metrics()

        
        pygame.display.flip()

        
        if pause_ms:
            pygame.time.wait(pause_ms)
        else:
            self.clock.tick(self.fps)

        return True

    def _draw_grid(self):
        """Draw grid lines"""
        if not self.show_grid:
            return

        for x in range(0, self.width + 1, self.cell_size):
            pygame.draw.line(
                self.screen,
                self.COLORS['grid'],
                (x, 0),
                (x, self.height),
                1
            )

        for y in range(0, self.height + 1, self.cell_size):
            pygame.draw.line(
                self.screen,
                self.COLORS['grid'],
                (0, y),
                (self.width, y),
                1
            )

    def _draw_static_entities(self):
        """Draw walls, shelves, and drop points"""
        for x in range(self.world.height):
            for y in range(self.world.width):
                cell_type = self.world.grid[x, y]

                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                if cell_type == 1:  
                    pygame.draw.rect(self.screen, self.COLORS['wall'], rect)
                elif cell_type == 2:  
                    pygame.draw.rect(self.screen, self.COLORS['shelf'], rect)
                    
                    text = self.font.render('S', True, (255, 255, 255))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

                    
                    inventory = self.world.shelf_inventory.get((x, y), 0)
                    inv_text = self.font_small.render(str(inventory), True, (255, 255, 255))
                    inv_rect = inv_text.get_rect(bottomright=(rect.right - 3, rect.bottom - 3))
                    self.screen.blit(inv_text, inv_rect)

                elif cell_type == 3:  
                    pygame.draw.rect(self.screen, self.COLORS['drop'], rect)
                    # Draw 'D' on drop point
                    text = self.font.render('D', True, (255, 255, 255))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

    def _draw_robots(self):
        """Draw robots with different colors based on carrying status"""
        for robot_id, (x, y) in self.world.robots.items():
            
            base_color = self.robot_colors[robot_id]

            
            if self.world.carrying[robot_id]:
                
                color = self.COLORS['robot_carrying']
            else:
                color = base_color

           
            rect = pygame.Rect(
                y * self.cell_size + 5,
                x * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10
            )

            
            center = rect.center
            radius = (self.cell_size - 10) // 2
            pygame.draw.circle(self.screen, color, center, radius)

           
            border_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.circle(self.screen, border_color, center, radius, 2)

            
            robot_num = robot_id.split('_')[1]
            text = self.font_small.render(robot_num, True, (255, 255, 255))
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)

            
            if self.world.carrying[robot_id]:
                box_rect = pygame.Rect(
                    rect.centerx - 5,
                    rect.top - 8,
                    10,
                    8
                )
                pygame.draw.rect(self.screen, (139, 69, 19), box_rect)  # Brown box
                pygame.draw.rect(self.screen, (0, 0, 0), box_rect, 1)  # Black border

    def _draw_metrics(self):
        """Draw metrics overlay at bottom of screen"""
        
        metrics_surface = pygame.Surface((self.width, self.metrics_height))
        metrics_surface.fill(self.COLORS['text_bg'][:3])
        metrics_surface.set_alpha(self.COLORS['text_bg'][3])
        self.screen.blit(metrics_surface, (0, self.height))

        
        metrics = self.world.get_metrics()

        
        lines = [
            f"Step: {metrics['step_count']}",
            f"Deliveries: {metrics['total_deliveries']}",
            f"Collisions: {metrics['collision_count']}",
            f"Efficiency: {metrics['deliveries_per_step']:.3f}",
        ]

        
        y_offset = self.height + 10
        for line in lines:
            text = self.font.render(line, True, self.COLORS['text'])
            self.screen.blit(text, (10, y_offset))
            y_offset += 25

        
        legend_x = self.width - 200
        legend_y = self.height + 10

        legend_title = self.font.render("Robots:", True, self.COLORS['text'])
        self.screen.blit(legend_title, (legend_x, legend_y))
        legend_y += 25

        for robot_id, color in self.robot_colors.items():
            robot_num = robot_id.split('_')[1]
            carrying = self.world.carrying[robot_id]

            
            pygame.draw.circle(
                self.screen,
                self.COLORS['robot_carrying'] if carrying else color,
                (legend_x + 10, legend_y + 8),
                8
            )

            
            status = " (carrying)" if carrying else ""
            text = self.font_small.render(f"Robot {robot_num}{status}", True, self.COLORS['text'])
            self.screen.blit(text, (legend_x + 25, legend_y))

            legend_y += 20

    def save_screenshot(self, filename: Optional[str] = None):
        """Save current frame as image"""
        if filename is None:
            import time
            filename = f"warehouse_{int(time.time())}.png"

        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved: {filename}")

    def close(self):
        """Clean up pygame"""
        pygame.quit()



if __name__ == "__main__":
    import random
    from world import World, STAY, UP, DOWN, LEFT, RIGHT, PICK, DROP_ACT

    print("Testing WarehouseRenderer...")
    print("Controls:")
    print("  ESC - Exit")
    print("  S - Save screenshot")
    print()

    
    world = World(height=12, width=12, n_robots=3)

    
    renderer = WarehouseRenderer(
        world,
        cell_size=60,
        fps=5,
        show_grid=True,
        show_metrics=True
    )

    
    running = True
    for step in range(200):
        if not running:
            break

       
        actions = {}
        for robot_id in world.robots:
            actions[robot_id] = random.randint(0, 6)

        
        rewards, info = world.step(actions)

       
        running = renderer.render()

        
        for robot_id, agent_info in info.items():
            if agent_info['picked']:
                print(f"✓ {robot_id} picked an item!")
            if agent_info['dropped']:
                print(f"✓ {robot_id} dropped an item!")
            if agent_info['collision']:
                print(f"✗ {robot_id} collided!")

        
        if step % 100 == 99:
            print(f"\n{'=' * 50}")
            print(f"Resetting environment...")
            print(f"Final metrics: {world.get_metrics()}")
            print(f"{'=' * 50}\n")
            world.reset()

    renderer.close()

    print("Done!")
