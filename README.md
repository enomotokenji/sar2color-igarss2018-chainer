# Image Translation Between Sar and Optical Imagery with Generative Adversarial Nets

This repository is an implementation of "Image Translation Between Sar and Optical Imagery with Generative Adversarial Nets".

## Setup
### Install required python libraries
```
pip install -r requirements.txt
```

### Download Palsar-Aster dataset (WIP)


## Training examples
You need set each parameters in a config file.  
```
CUDA_VISIBLE_DEVICES=0 python train_pix2pix.py --config_path configs/config_pix2pix.yml --results_dir results/pix2pix
```
If you want to resume the training from snapshot, use `--snapshot` option.

* pretrained model (WIP)

## Evaluation examples
```
CUDA_VISIBLE_DEVICES=0 python evaluations/test.py --results_dir results/test_pix2pix --config_path results/pix2pix/config_pix2pix.yml --gen_model results/pix2pix/Generator_<iterations>.npz
```

## License
Academic use only.
