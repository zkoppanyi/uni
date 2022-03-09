## Structure from Motion (SfM)
- post processed
- unordered image set (e.g. photo shots)
- application: mapping
- feature matching
- pose prior available
- refine with bundle adjustment

## Visual SLAM
- real-time
- ordered image set (e.g. video)
- application: navigation/robotics
- feature tracking
- no pose prior
- refing with pose graph, loop closures
- data source for SLAM is typically not only images, but LiDAR, etc.
