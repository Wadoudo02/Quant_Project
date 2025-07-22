import numpy as np
import matplotlib.pyplot as plt

def estimate_circle_area(num_points: int):
    """
    Estimates the area of a circle using a Monte Carlo simulation.

    Args:
        num_points: The total number of random points to generate.

    Returns:
        A tuple containing:
        - The estimated area of the circle.
        - The x and y coordinates of points inside the circle.
        - The x and y coordinates of points outside the circle.
    """
    # Let the square have side a = 2, so it extends from -1 to 1 on both axes.
    # The inscribed circle is centered at (0,0) with radius r = 1.
    a = 2
    r = a / 2

    # Area of the square is a^2
    area_square = a**2

    # Generate random points within the square.
    # x and y coordinates are between -r and r (-1 and 1).
    x_coords = np.random.uniform(-r, r, num_points)
    y_coords = np.random.uniform(-r, r, num_points)

    # Calculate the distance from the center (0,0) squared.
    distance_sq = x_coords**2 + y_coords**2

    # Check which points fall inside the circle (distance <= radius).
    is_inside = distance_sq <= r**2

    # Count the number of points inside the circle.
    points_inside_count = np.sum(is_inside)

    # Estimate the area of the circle.
    estimated_area = (points_inside_count / num_points) * area_square

    # Separate points for plotting
    points_inside_x = x_coords[is_inside]
    points_inside_y = y_coords[is_inside]
    points_outside_x = x_coords[~is_inside]
    points_outside_y = y_coords[~is_inside]

    return (
        estimated_area,
        (points_inside_x, points_inside_y),
        (points_outside_x, points_outside_y),
    )


# --- Part 1: Show how approximation improves as n gets larger ---
print("--- Improving Approximation with More Points ---")

# The true area of a circle with radius 1 is π * r^2 = π
true_area = np.pi
print(f"True Area of Circle (π * 1^2): {true_area:.6f}\n")

num_points_list = [100, 1_000, 10_000, 100_000, 1_000_000] #, 1_000_000_000]

for n in num_points_list:
    est_area, _, _ = estimate_circle_area(n)
    error = abs((est_area - true_area) / true_area) * 100
    print(
        f"Points (n) = {n:<10} | "
        f"Estimated Area = {est_area:.6f} | "
        f"Error = {error:.4f}%"
    )


# --- Part 2: Generate a plot showing the points ---
print("\n--- Generating Plot ---")

# Use a moderate number of points for clear visualization
n_for_plot = 5000
_, points_inside, points_outside = estimate_circle_area(n_for_plot)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal', adjustable='box')

# Plot the points
ax.scatter(
    points_inside[0], points_inside[1], color="#3377FF", s=5, label="Inside Circle"
)
ax.scatter(
    points_outside[0], points_outside[1], color="#FF5533", s=5, label="Outside Circle"
)

# Draw the circle and the square for reference
radius = 1.0
circle = plt.Circle((0, 0), radius, color='black', fill=False, linewidth=2)
square = plt.Rectangle((-radius, -radius), 2*radius, 2*radius, color='black', fill=False, linewidth=2)

ax.add_patch(circle)
ax.add_patch(square)

# Formatting
plt.title(f"Monte Carlo Estimation of Circle Area (n = {n_for_plot})")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

# Show the plot
plt.show()

print("Plot has been generated and displayed.")