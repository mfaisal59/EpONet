# Exploiting Geometric Constraints on Dense Trajectories for Motion Saliency

The existing approaches for salient motion segmentation are unable to explicitly learn geometric cues and often give false detections on prominent static objects. We exploit multiview geometric constraints to avoid such mistakes. To handle nonrigid background like sea, we also propose a robust fusion mechanism between motion and appearance-based features. We find dense trajectories, covering every pixel in the video, and propose trajectory-based epipolar distances to distinguish between background and foreground regions. Trajectory epipolar distances are data-independent and can be readily computed given a few features' correspondences in the images. We show that by combining epipolar distances with optical flow, a powerful motion network can be learned.

![alt text](https://github.com/mfaisal59/EpONet/blob/master/images/flowDiagram.png)

# Instructions
### Epipolar Score Computation
The epipolar score computation code can be downloaded from [link](https://github.com/mfaisal59/EpipolarScore). 

[EpO] (https://drive.google.com/file/d/1LxIyiHPoR5gIjs4bsZMtktsfPqJ1CZ8B/view?usp=sharing)
[EpO+ Trained Network] (https://drive.google.com/file/d/1tBfS5JTrx5bqaQaF5kwxhTg1Zc1_E2iR/view?usp=sharing)
[DeepLab] (https://drive.google.com/file/d/18u8lIiO4i1QD65XNvZI-mxjUrrRbwPrs/view?usp=sharing)

#####Pre-Computed Results
[EpO] (https://drive.google.com/drive/folders/1A2ewOKvLwZy0A83AZEC9XivZPNxm0PJB?usp=sharing)
[Epo+] (https://drive.google.com/drive/folders/1gvMmAarNLfru7IVYkzfXuekhCMjcjYnO?usp=sharing)

[pre-Computed Epipolar Score] (https://drive.google.com/drive/folders/1gvMmAarNLfru7IVYkzfXuekhCMjcjYnO?usp=sharing)
[pre-Computed Optical Flow] (https://drive.google.com/drive/folders/1gvMmAarNLfru7IVYkzfXuekhCMjcjYnO?usp=sharing)
[pre-Computed Motion Images]
