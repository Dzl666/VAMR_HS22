# Mini Project of Visual Odometry for Lecture "Visual Algorithm of Mobile Robots"

## Baisc information

* project credit: UZH RPG Lab
* code author: Zilong Deng
* environment: Matlab 2017a and above
* datasets: parking(simple), KITTI(hard), malaga-urban(hard&large)
* camera: single calibrated RGB camera

## Structure

none

## Pipeline

### Bootstarp (& Reboot)

1. Pick two non-adjecent images with proper baseline
2. Use a Harris detector to extract keypoints(kpts) in the first image
3. Use a KLT tracker to find corresponding featrures in the second image
4. Calculate the Fundemantal matrix with RANSAC
5. Generate the landmarks using the 2D features and poses of the camera on two images

### Feature Tracking

1. Given all features in previous image
2. sing KLT to track corresponding features in the new image

### Pose Estimation & Optimization

1. Given features in current image and corresponding landmarks linked to features in previous pictures
2. Use P3P with RANSAC to estimate the pose of the camera in current image

### Triangulate new landmarks

### Landmarkes update

## Dashboard

* Landmarks detected in current image
* Trajectory
* Number of landmarks in the past 20 images
