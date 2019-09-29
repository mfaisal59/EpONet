# Exploiting Geometric Constraints on Dense Trajectories for Motion Saliency

The existing approaches for salient motion segmentation are unable to explicitly learn geometric cues and often give false detections on prominent static objects. We exploit multiview geometric constraints to avoid such mistakes. To handle nonrigid background like sea, we also propose a robust fusion mechanism between motion and appearance-based features. We find dense trajectories, covering every pixel in the video, and propose trajectory-based epipolar distances to distinguish between background and foreground regions. Trajectory epipolar distances are data-independent and can be readily computed given a few features' correspondences in the images. We show that by combining epipolar distances with optical flow, a powerful motion network can be learned.

![alt text](https://github.com/mfaisal59/EpONet/blob/master/images/flowDiagram.png)

The is a public implementation of our WACV 2020 paper on Exploiting Geometric Constraints on Dense Trajectories for Motion Saliency. This repository contains testing code and trained models.

1. Installations:

Our implementation is based on the Torch framework (http://torch.ch). It depends on the lua/torch packages "nnx", "rnn" and "extracunn". The first can be installed with

	luarocks install nnx 

The other two are installed with 
	
	git clone https://github.com/Element-Research/rnn; cd rnn; luarocks make rocks/rnn-scm-1.rockspec
	git clone https://github.com/viorik/extracunn.git; cd extracunn; luarocks make 

You will aslo need to have a relatively recently version of MATLAB, for computation of Optical Flow and Epipolar Score. 


###. Instructions:


###### 1) Epipolar Score Computation

The epipolar score computation code and instructions can be downloaded from [link](https://github.com/mfaisal59/EpipolarScore). 

###### 2) Clone the repository
	
```
git clone https://github.com/mfaisal59/EpONet.git
```

###### 3) Download Trained Models:

```
cd EpONet/
bash ./models/download_models.sh
#These command will populate the `./models/` folder with trained models.
```

###### 4) Download our pre-computed Epipolar Score, Optical Flow, motion Images and JPEGImages for two test sequences from DAVIS Dataset.

```
cd EpONet/
bash ./DAVIS_Dataset/download_data.sh
#These command will populate the `./DAVIS_Dataset/` folder.
```

###### 5) Test EpO (Motion Network)

```
th testDAVIS_motion.lua -gpu $GPU_ID -model $MODEL_NAME
#modify the path in testDAVIS_motion.lua & segmentDAVIS_motion.lua script.
#where GPU_ID stands for the index of GPU, and MODEL_NAME is motion model i.e. DAVISFineTuned.dat
```

###### 6) Test EpO+ (Fusion Network)

```
th testDAVIS_Fusion.lua -gpu $GPU_ID -model -memoryModel $MODEL_NAME -motionModel DAVISFineTuned.dat
#modify the path in testDAVIS_Fusion.lua & segmentFrame_Fusion.lua script.
#where GPU_ID stands for the index of GPU, and MODEL_NAME is fusion model i.e. fusionBestNet_DAVIS_2016.dat
```
		
### Pre-Computed Results
[EpO] (https://drive.google.com/drive/folders/1A2ewOKvLwZy0A83AZEC9XivZPNxm0PJB?usp=sharing)

[Epo+] (https://drive.google.com/drive/folders/1gvMmAarNLfru7IVYkzfXuekhCMjcjYnO?usp=sharing)
