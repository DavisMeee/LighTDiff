# [MICCAI2024#224] LighTDiff: Surgical Endoscopic Image Low-Light Enhancement with T-Diffusion


**LighT Architecture Diffusion Model For Low-light Image Enhancement**

## Schematics
![MainFrame](Schematric/Schematric.png)
## Results
![Visualization](Examples/result1.png)
![Visualization](Examples/result2.png)
![Visualization](Examples/result3.png)
## Pre-installation
```Install step
conda create -n lightdiff python=3.10
conda activate lightdiff
conda install pytorch==2.0.1 torchvision torchaudio cudatoolkit -c pytorch
cd BasicSR-light
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
cd ../lightdiff
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
