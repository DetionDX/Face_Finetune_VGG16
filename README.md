# Face_Finetune_VGG16
Rutgers University mid-term project ECE 539: Advanced Topics in DSP - Deep Learning for Biometrics (Spring 2017)

download the pretrained vgg16_weights.npz model from: http://www.cs.toronto.edu/~frossard/post/vgg16/

download the lfw data and pair.txt from: http://vis-www.cs.umass.edu/lfw/

preprocessing crop the images using MTCNN: https://kpzhang93.github.io/MTCNN_face_detection_alignment/



(1) To run the result directly using the pretrained VGG16: 

$ python lfw.py 

tips: you should modify the image_path in the code "lfw.py" to your local folder stored the lfw images.




(2) Finetuning:

This work is finetuning the pretrained weights VGG16 on LFW PairsDevtrain.txt using siamese network. 

vgg16_siamese.py defined the network framework.

finetune_lfw.py gives the main code.

To run these codes, you should set the lfw dataset path and PairsDevtrain.txt, PairsDevtest.txt and pairs.txt properly in finetune_lfw.py. Also, you may want to adjust the margin in vgg16_siamese.py codes.

Problem:

For now, the only problem I noticed is after several epochs, the loss goes to nan. I am still working on it and see what causes that. For some reason, people said it is based on the cuda driver and cudnn version. However, I am not sure, I will fix this in the future.
