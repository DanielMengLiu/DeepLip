# Update 2023-05-09
[important] Please turn to our latest work AVLip accepted by ICASSP2023. https://github.com/DanielMengLiu/AudioVisualLip
A lot of optimization is done in AVLip including (but not limitted):
* better performed systems
* the audio-/visual-only structures
* parallel data processing and score decision
* new datasets and compressed in .mp4 format
* data augmentation

# DeepLip
deep-learning based audio-visual lip bometrics

ASRU paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9688240
![image](https://user-images.githubusercontent.com/45690014/162400266-9fa40604-712e-47bb-b22f-16b757e4ebcd.png)

trained models: https://drive.google.com/drive/folders/1IalsNtmDH-qFnfgmn_O92J1MUHCaQepl?usp=sharing
(similar performance as the ASRU paper but not exactly the same)

# Author
 Meng Liu, Chang Zeng, Hanyi Zhang
 
 e-mail: liumeng2017@tju.edu.cn
 
# Content
This work has been submitted to Interspeech2021. It introduces a deep-learning based audio-visual lip biometrics framework, illustrated as the above figure. Since the audio-visual lip biometrics study has been hindered by the lack of appropriate and sizeable database, this work presents a moderate baseline database using public lip datasets, as well as the baseline system. 

Audio-visual lip biometrics is interesting. Different from other audio-visual speaker recognition methods, it leverages the multimodal information from the audible speech and visual speech (i.e., lip movements). Many work hasn't been explored in this area. We will update the code and resource as the advancement of our process.

# Have Done
* establish a public DeepLip database as well as a well performed baseline system
* prove the feasibility of deep-learning based audio-visual lip biometrics
* show the complementary power of fusing the audible speech and visual speech 

# To Do List
* complex multimodal fusion methods
* compared with other audio-visual speaker recognition methods
* prove the robustness of spoof and noisy enviroments
* text-dependent audio-visual lip biometrics
* collect large audio-visual lip database

# Cite
@inproceedings{liu2021deeplip,
  title={DeepLip: A Benchmark for Deep Learning-Based Audio-Visual Lip Biometrics},
  author={Liu, Meng and Wang, Longbiao and Lee, Kong Aik and Zhang, Hanyi and Zeng, Chang and Dang, Jianwu},
  booktitle={2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={122--129},
  year={2021},
  organization={IEEE}
}
