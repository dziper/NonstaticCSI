import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

class MobilitySimulation:
    def __init__(self, x_range, y_range,
                 road_width=10,
                 total_steps=100,
                 max_acceleration=1.0,
                 max_velocity=5.0,
                 finish_distance=5.0,
                 save_folder="simulation_output",
                 file_name="trajectory.csv"):
        """
        Initialize the mobility simulation for a road defined by two points

        Args:
        - x_range: [start_x, end_x] of the road
        - y_range: [start_y, end_y] of the road
        - road_width: Width of the road (perpendicular to road line)
        - total_steps: Number of time steps to simulate
        - max_acceleration: Maximum random acceleration per step
        - max_velocity: Maximum velocity magnitude
        - finish_distance: Distance threshold to consider the end point reached
        - save_folder: Folder to save the trajectory data (CSV)
        - file_name: Name of the CSV file to save the trajectory
        """
        self.start_point = np.array([x_range[0], y_range[0]])
        self.end_point = np.array([x_range[1], y_range[1]])

        # Calculate road characteristics
        self.road_vector = self.end_point - self.start_point
        self.road_length = np.linalg.norm(self.road_vector)
        self.road_direction = self.road_vector / self.road_length

        # Perpendicular vector for road width
        self.perpendicular_vector = np.array([-self.road_direction[1], self.road_direction[0]])

        self.road_width = road_width
        self.total_steps = total_steps
        self.max_acceleration = max_acceleration
        self.max_velocity = max_velocity
        self.finish_distance = finish_distance

        # Initialize position at the start point
        self.position = self.start_point.copy()

        # Initialize velocity
        self.velocity = np.random.uniform(-max_velocity, max_velocity, 2)

        # Store trajectory for visualization
        self.trajectory = [self.position.copy()]
        self.velocities = [self.velocity.copy()]

        # Flag to track if the end point has been reached
        self.end_point_reached = False

        # Save folder setup
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # File name for saving the trajectory
        self.file_name = file_name


        self.loaded_trajectories = []

    def _project_onto_road(self, point):
        """
        Project a point onto the road line
        """
        point_vector = point - self.start_point
        road_proj = np.dot(point_vector, self.road_direction)
        road_proj = np.clip(road_proj, 0, self.road_length)

        return self.start_point + road_proj * self.road_direction

    def _distance_from_road(self, point):
        """
        Calculate perpendicular distance from a point to the road line
        """
        proj_point = self._project_onto_road(point)
        return np.linalg.norm(point - proj_point)

    def _handle_boundary_collision(self):
        """
        Randomly change velocity direction if hitting road boundaries
        """
        dist_from_road = self._distance_from_road(self.position)

        if dist_from_road > self.road_width / 2:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.linalg.norm(self.velocity)

            new_velocity = speed * np.array([
                np.cos(angle) * np.linalg.norm(self.road_direction),
                np.sin(angle) * np.linalg.norm(self.road_direction)
            ])

            self.velocity = new_velocity

            # Adjust position to be within road boundaries
            proj_point = self._project_onto_road(self.position)
            correction = (self.road_width / 2) * np.sign(np.dot(self.position - proj_point, self.perpendicular_vector)) * self.perpendicular_vector
            self.position = proj_point + correction

    def _avoid_backward_movement(self):
        """
        Ensure that the vehicle always moves towards the end point, and does not go backward.
        """
        # Calculate the dot product of current velocity and road direction to check if moving backwards
        dot_product = np.dot(self.velocity, self.road_direction)

        # If moving away from the end point (negative dot product), reverse direction towards the end point
        if dot_product < 0:
            # Reorient velocity to always move towards the end point
            self.velocity = np.abs(self.velocity) * self.road_direction  # Ensure movement is in the correct direction

    def _check_end_point_reached(self):
        """
        Check if the end point has been reached
        """
        distance_to_end = np.linalg.norm(self.position - self.end_point)
        if distance_to_end <= self.finish_distance:
            self.end_point_reached = True

    def simulate(self):
        """
        Run the mobility simulation
        """
        for _ in range(self.total_steps):
            acceleration = np.random.uniform(-self.max_acceleration, self.max_acceleration, 2)

            # Update velocity based on acceleration
            self.velocity += acceleration

            # Limit velocity magnitude
            velocity_mag = np.linalg.norm(self.velocity)
            if velocity_mag > self.max_velocity:
                self.velocity = self.velocity / velocity_mag * self.max_velocity

            # Update position
            self.position += self.velocity

            # Handle boundary collisions
            self._handle_boundary_collision()

            # Avoid going backward
            self._avoid_backward_movement()

            # Check if end point has been reached
            self._check_end_point_reached()

            # Track trajectory
            self.trajectory.append(self.position.copy())
            self.velocities.append(self.velocity.copy())

            # Stop simulation if end point is reached
            if self.end_point_reached:
                break

        # Save trajectory data
        self.save_trajectory()

        return np.array(self.trajectory), np.array(self.velocities)

    def save_trajectory(self):
        """
        Save the trajectory data to a CSV file in the specified folder
        """
        file_path = os.path.join(self.save_folder, self.file_name)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'X Position', 'Y Position'])
            for step, position in enumerate(self.trajectory):
                writer.writerow([step, position[0], position[1]])
    def load_saved_trajectory(self,directory):

        # Create a list to store trajectories
        self.loaded_trajectories = []

        # Read and sort all trajectory filenames
        trajectory_files = sorted(
            [f for f in os.listdir(directory) if f.startswith('trajectory_') and f.endswith('.csv')],
            key=lambda x: int(x.split('_')[1].split('.')[0])  # Sort by numeric part of the filename
        )

        # Read each trajectory file and append it to the list
        for filename in trajectory_files:
            filepath = os.path.join(directory, filename)
            trajectory = pd.read_csv(filepath).to_numpy()  # Adjust delimiter if needed
            trajectory = trajectory[:, 1:]
            self.loaded_trajectories.append(trajectory)

    def plot_trajectory(self):
        """
        Visualize the mobility route
        """
        trajectory = np.array(self.trajectory)
        plt.figure(figsize=(12, 6))

        plt.plot([self.start_point[0], self.end_point[0]], [self.start_point[1], self.end_point[1]], 'k-', label='Road Line')

        road_start_left = self.start_point + (self.road_width / 2) * self.perpendicular_vector
        road_start_right = self.start_point - (self.road_width / 2) * self.perpendicular_vector
        road_end_left = self.end_point + (self.road_width / 2) * self.perpendicular_vector
        road_end_right = self.end_point - (self.road_width / 2) * self.perpendicular_vector

        plt.plot([road_start_left[0], road_end_left[0]], [road_start_left[1], road_end_left[1]], 'r--', label='Road Boundary')
        plt.plot([road_start_right[0], road_end_right[0]], [road_start_right[1], road_end_right[1]], 'r--')

        plt.plot(trajectory[:, 0], trajectory[:, 1], label='Mobility Path')

        plt.scatter(self.start_point[0], self.start_point[1], color='green', s=100, label='Start Point')
        plt.scatter(self.end_point[0], self.end_point[1], color='red', s=100, label='End Point')

        plt.title('Random Mobility Route with Backward Movement Avoidance')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    def plot_loaded_trajectories(self,savedir=False,num_paths=5):
        """
        Plot multiple trajectories in a background-free picture, each with a random color.

        Parameters:
        trajectories (list of np.ndarray): List of trajectories, where each trajectory is a 2D NumPy array.
        """
        plt.figure()

        # Plot each trajectory with a random color
        for trajectory in self.loaded_trajectories[0:num_paths]:
            trajectory = np.array(trajectory)
            color = np.random.rand(3, )  # Generate a random RGB color
            plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2)


        # # Background-free settings
        # plt.axis('off')  # Remove axes
        # plt.axis('equal')  # Keep aspect ratio
        # plt.tight_layout()  # Remove unnecessary padding
        if(savedir):
            plt.savefig(savedir, format='png',dpi=300)
        else:
            plt.show()





# Example usage
sim = MobilitySimulation(
    x_range=[0,402.73],
    y_range=[135.97,138.33],
    road_width=10,
    total_steps=5000,
    max_acceleration=1,
    max_velocity=2.0,
    finish_distance=5.0,
    save_folder="simulation_output",
    file_name="custom_trajectory2.csv"
)
# trajectory, velocities = sim.simulate()
# sim.plot_trajectory()

# print("Trajectory Start:", trajectory[0])
# print("Trajectory End:", trajectory[-1])
# print("Distance to End Point:", np.linalg.norm(trajectory[-1] - sim.end_point))
# print("Total Trajectory Length:", len(trajectory))


trajectory_dir = f"C:\\Users\ibrahimkilinc\Documents\ECE257_Project\\NonstaticCSI\Dataset\path_trajectories"
sim.load_saved_trajectory(trajectory_dir)

sim_res_path = f"C:\\Users\ibrahimkilinc\Documents\ECE257_Project\\NonstaticCSI\simulation_results"
paths_image_dir = os.path.join(sim_res_path, "paths_image.png")

sim.plot_loaded_trajectories(paths_image_dir,num_paths = 2)
