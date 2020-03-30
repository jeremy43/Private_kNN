We also provide a pytorch implementation of PATE-2018 https://arxiv.org/abs/1802.08908
Instructions of PATE framework is 
1. `python train_teacher.py` to train k teacher models
2. `python pate.py` to train a student model in the public domain
3. Using the answered pseudo-labels in step 2 together with the corresponding features to do semi-supervised training with UDA

Note that the accuracy reported in paper PATE2 and private-kNN are both based on the semi-supervised training. 
