# Exploiting Geometric Constraints on Dense Trajectories for Motion Saliency

The existing approaches for salient motion segmentation are unable to explicitly learn geometric cues and often give false detections on prominent static objects. We exploit multiview geometric constraints to avoid such mistakes. To handle nonrigid background like sea, we also propose a robust fusion mechanism between motion and appearance-based features. We find dense trajectories, covering every pixel in the video, and propose trajectory-based epipolar distances to distinguish between background and foreground regions. Trajectory epipolar distances are data-independent and can be readily computed given a few features' correspondences in the images. We show that by combining epipolar distances with optical flow, a powerful motion network can be learned.

![alt text](https://github.com/mfaisal59/EpONet/blob/master/images/flowDiagram.png)

# Instructions
### Epipolar Score Computation
The epipolar score computation code can be downloaded from [link](https://github.com/mfaisal59/EpipolarScore)