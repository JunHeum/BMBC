# BMBC

Junheum Park,
Keunsoo Ko, 
Chul Lee,
and Chang-Su Kim

Official PyTorch Code for **"BMBC: Bilateral Motion Estimation with Bilateral Cost Volume for Video Interpolation"** [[paper]](https://arxiv.org/abs/2007.12622)

### Requirements
- PyTorch 1.3.1 (Other versions can cause different results)
- CUDA 10.0
- CuDNN 7.6.5
- python 3.6

### Installation
Create conda environment:
```
    $ conda create -n BMBC python=3.6 anaconda
    $ conda activate BMBC
    $ pip install opencv-python
    $ conda install pytorch==1.3.1 torchvision cudatoolkit=10.0 -c pytorch
```
Download repository:
```
    $ git clone https://github.com/JunHeum/BMBC.git
```
Download [pre-trained model](https://drive.google.com/u/0/uc?export=download&confirm=JG-C&id=1YIxJguzgFp0t7iLHw1bKLrv7JQrc4lMG) parameters:
```
    $ unzip BMBC_Weights.zip
```
### Usage
Generate an intermediate frame at `t=0.5` on your pair of frames:
```
    $ python run.py --first images/im1.png --second images/im3.png --output images/im2.png
```    
Generate an intermediate frame at arbitrary time `t`:
```
    $ python run.py --first images/im1.png --second images/im3.png --output images/im2_025.png --time_step 0.25 
```
### Citation
Please cite the following paper if you feel this repository useful.
```
    @inproceedings{BMBC,
        author    = {Junheum Park, Keunsoo Ko, Chul Lee,and Chang-Su Kim}, 
        title     = {BMBC: Bilateral Motion Estimation with Bilateral Cost Volume for Video Interpolation}, 
        booktitle = {European Conference on Computer Vision},
        year      = {2020}
    }
```
### License
See [MIT License](https://github.com/JunHeum/BMBC/blob/master/LICENSE)
