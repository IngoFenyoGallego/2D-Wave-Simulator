import pygame
import numpy as np

# Constants for the simulation
GRID_SIZE = (200, 200)    # (rows, columns) = (height, width)
WINDOW_SIZE = (1000, 1000)  # (width, height) in pixels
BACKGROUND_COLOR = (0, 0, 0)  # Black background for the visualization

WAVE_SPEED = 0.15  # Wave propagation coefficient (c²Δt² in discrete wave equation)
DAMPING = 0.988    # Makes ripples slowly fade out
SCALE = 4.5        # Visualization scale

# Barrier parameters
BARRIER_X = GRID_SIZE[1] // 2          # Vertical barrier down the middle (columns)
BARRIER_THICKNESS = 3                  
BARRIER_GAP = 6                        
BARRIER_GAP_SEP = 26                   
BARRIER_COLOR = np.array([139, 69, 19])  # Brown (RGB)


def add_bump(u_current, grid_x, grid_y):
    """Add a rock-drop impulse (Mexican hat) to spawn tight, uniform concentric ripples."""
    for dy in range(-5, 6):      
        for dx in range(-5, 6):
            x = grid_x + dx
            y = grid_y + dy
            if 0 <= x < GRID_SIZE[1] and 0 <= y < GRID_SIZE[0]:
                r2 = dx * dx + dy * dy
                # Mexican hat function
                u_current[y, x] += 0.8 * (1 - 0.4 * r2) * np.exp(-0.22 * r2)


def laplacian(u):
    """Compute the discrete Laplacian of a 2D grid."""
    laplace = np.zeros_like(u)
    laplace[1:-1, 1:-1] = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
    return laplace


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Interactive 2D Ripple Tank")
    clock = pygame.time.Clock()

    # Create the wave field
    u_prev = np.zeros(GRID_SIZE)     # Displacement at t-1
    u_current = np.zeros(GRID_SIZE)  # Displacement at t
    u_next = np.zeros(GRID_SIZE)     # Displacement at t+1

    # Build barrier mask with a central gap to observe diffraction
    barrier = np.zeros(GRID_SIZE, dtype=bool)
    mid_y = GRID_SIZE[0] // 2
    gap1_center = mid_y - BARRIER_GAP_SEP // 2
    gap2_center = mid_y + BARRIER_GAP_SEP // 2
    gap_half = BARRIER_GAP // 2

    def apply_gap(center_y):
        start = max(center_y - gap_half, 0)
        end = min(center_y + gap_half, GRID_SIZE[0])
        return start, end

    g1_start, g1_end = apply_gap(gap1_center)
    g2_start, g2_end = apply_gap(gap2_center)

    for dx in range(BARRIER_THICKNESS):
        x = BARRIER_X + dx
        if 0 <= x < GRID_SIZE[1]:
            # Fill barrier everywhere except the two gap ranges
            barrier[:g1_start, x] = True
            barrier[g1_end:g2_start, x] = True
            barrier[g2_end:, x] = True

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # Convert mouse position to grid coordinates
                grid_x = int(mouse_x / (WINDOW_SIZE[0] / GRID_SIZE[1]))
                grid_y = int(mouse_y / (WINDOW_SIZE[1] / GRID_SIZE[0]))
                add_bump(u_current, grid_x, grid_y)

        # Compute the Laplacian of the current wave field
        laplace = laplacian(u_current)

        # Update the wave field using the wave equation
        u_next[1:-1, 1:-1] = 2 * u_current[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + WAVE_SPEED * laplace[1:-1, 1:-1]

        # Apply damping
        u_next *= DAMPING

        # Enforce boundary conditions (set edges and barrier to zero)
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0
        u_next[barrier] = 0

        # Rotate the grids (u_prev <- u_current, u_current <- u_next)
        u_prev = u_current.copy()
        u_current = u_next.copy()

        screen.fill(BACKGROUND_COLOR)

        # Scale and clip wave heights for visualization
        normalized = np.clip(u_current * SCALE, -1.0, 1.0)

        # Map to grayscale: 128 = zero, 255 = max positive, 1 = max negative
        gray = np.clip(128 + normalized * 127, 0, 255).astype(np.uint8)

        # Build RGB grid (blue tint)
        rgb_grid = np.stack([gray * 0.3, gray * 0.5, gray], axis=-1)
        rgb_grid[barrier] = BARRIER_COLOR
        rgb_grid = np.transpose(rgb_grid, (1, 0, 2))

        surface = pygame.surfarray.make_surface(rgb_grid)
        surface = pygame.transform.scale(surface, WINDOW_SIZE)
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Cap render FPS (physics step is still per-iteration fixed)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()