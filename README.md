# Face_Finetune_VGG16
Rutgers University mid-term project ECE 539: Advanced Topics in DSP - Deep Learning for Biometrics (Spring 2017)

download the pretrained vgg16_weights.npz model from: http://www.cs.toronto.edu/~frossard/post/vgg16/

download the lfw data and pair.txt from: http://vis-www.cs.umass.edu/lfw/

preprocessing crop the images using MTCNN: https://kpzhang93.github.io/MTCNN_face_detection_alignment/

To run the result directly using the pretrained VGG16: 

$ python lfw.py 

tips: you should modify the image_path in the code to your local folder stored the lfw images.
