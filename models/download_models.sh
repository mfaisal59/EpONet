#!/bin/bash

#Download the pre-trained models

# function copied from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# Download EpO (motion only network)
fileid=1LxIyiHPoR5gIjs4bsZMtktsfPqJ1CZ8B
filename=EpO.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"

# Download the DeepLab (Appearance Network)
fileid=18u8lIiO4i1QD65XNvZI-mxjUrrRbwPrs
filename=DeepLab.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"

# Download the EpO+ (our Fusion Network)
fileid=1tBfS5JTrx5bqaQaF5kwxhTg1Zc1_E2iR
filename=EpOPlus.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"