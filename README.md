[//]: # (Image References)




# Image-Captioning

This Repository contains the Pytorch implementation of the research paper [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) with improvemnts in the decoder part as we integrate BERT context vectors into training instead of LSTMs as used in the original paper.

Parts of this codes in this project has been taken from the following repositries:
1. https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py
2. https://github.com/ajamjoom/Image-Captions/blob/master/data_loader.py
3. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

<a href="url"><img src="https://raw.githubusercontent.com/AbdurNawaz/Image-Captioning-with-BERT/master/images/cat.png" align="centre" height="350" width="350" ></a> 
<a href="url"><img src="https://raw.githubusercontent.com/AbdurNawaz/Image-Captioning-with-BERT/master/images/duck.png" align="right" height="350" width="350" ></a> 

__output__: a large black cat is sitting infron of the tv  &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp;  __output__: a duck floating the water near a tree branch

This project can be broken down into the following scripts:

__processData.py__: Running this file will pre process the COCO dataset and store it in `vocab.pkl` which can be used later.

__dataloader.py__ : This script will create the Dataset class required for the Pytorch model with applying different transformation to the images.

__endocer.py__: This script contains the `Encoder` class which is a Resnet 101 network with thw last pooling and linear layer removed.

__decodrder.py__: This 
