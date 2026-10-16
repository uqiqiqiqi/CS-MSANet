# CS-MSANet

## Train
```bash
python train.py --dataset Synapse --cfg configs/cswin_tiny_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR --img_size 224 --base_lr 0.05 --batch_size 24
```
## test
```bash
python test.py --dataset Synapse --cfg configs/cswin_tiny_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```
## datasets
```bash
he ISIC18 dataset can be found here {https://challenge.isic-archive.com/data/}.
The ACDC dataset can be found here {https://www.creatis.insa-lyon.fr/Challenge/acdc/}.
The Synapse dataset can be found here {https://www.synapse.org/#!Synapse:syn3193805/wiki/217789}.
```
