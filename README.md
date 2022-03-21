# Visual Inertial SLAM
## The project combines the data from IMU, Stereo camera to implement SLAM algorithm and uses Extended Kalman filter to plot the trajectory of the robot.
The project uses 2 files.
1. pr3_utils.py:
   - This file has basic functions such as:
     - a) load_data(file_name): This would return the essential parameters needed for the project.
     - b) visualize_trajectory(): This function is used to visualize the trajectory and world frame landmarks.
2. main_2.py:
   - This file imports functions from pr3_utils.py. Inside the main function, an object of datatype Visual_SLAM is created.
   - The member functions are the following:
     - a) homogenize(): Converts a vector to its homogenous form.
     - b) dehomogenize(): Dehomogenizes a given vector
     - c) skew_matrix(): Converts a vector to its skew matrix form
     - d) twist_matrix(): Converts a the generalized velocity vector to a twist matrix.
     - e) dead_reckoning(): Calculates the pose (only prediction) and plots the trajectory of the vehicle using the IMU data alone.
     - f) prior(): For time t=0, it calculates the necessary parameters.
     - g) mapping(): Using the observation model (i.e images) it computes the coordinates of the land marks in the world frame based on
                   the pose of the vehicle.
     - h) mapping_only(): Using the observation model (i.e images) it computes the coordinates of the land marks in the world frame based on
                   the pose of the vehicle. The pose of the vehicle is predicted but not updated.
     - i)prediction(): Computes the pose and calls mapping() and using the coordinates updates the pose of the vehicle. 

Link to [report](https://drive.google.com/file/d/1FJ0n4TsYZ6yMklSwxHH-jZkU-1n1WN69/view?usp=sharing)