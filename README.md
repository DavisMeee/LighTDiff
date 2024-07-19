# [MICCAI2024] LighTDiff: Surgical Endoscopic Image Low-Light Enhancement with T-Diffusion
## Update
**[7/20/2024] Fixed some bugs.**

**[5/13/2024] Our work got early accepted by MICCAI2024!**

**[5/17/2024] Our code is now available!**

## Schematics
![MainFrame](Schematric/Schematric.png)
## Results
![Visualization](Examples/result1.png)
![Visualization](Examples/result3.png)
## Pre-installation
```Install step
conda create -n lightdiff python=3.10
conda activate lightdiff
conda install pytorch==2.0.1 torchvision torchaudio cudatoolkit -c pytorch
cd BasicSR-light
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
cd ../LighTDiff
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
```

## Test
```
python lightdiff/train.py -opt configs/test.yaml
```
## Train
```
python lightdiff/train.py -opt configs/train.yaml
```
