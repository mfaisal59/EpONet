#!/bin/bash

# function copied from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# Download JPEGImages (two Sequences - Camel & Car-Roundabout)
fileid=1tYtkCbhYNZk0O1TDayFBYOSyMviwfcoO
filename=JPEGImages.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"

# Download Epipolar Score 
fileid=1KcxHUk1TCPCnWavXfId7lQPJXpO_dFd2
filename=Epipolar.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"

# Download Optical Flow
fileid=15yU5qsyuyq691MFZ8OE38IMunrwitK8S
filename=OpticalFlow.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"

# Download Motion Images
fileid=1LoTiOcsLsTqeNVr4rLWRAG1ng2p-Lh1e
filename=motionImages.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"