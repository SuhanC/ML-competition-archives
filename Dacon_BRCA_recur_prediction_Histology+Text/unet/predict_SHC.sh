#!/bin/bash
mkdir -p /home/suhan/data/Dacon_BRCA/train_segmented/
mkdir -p /home/suhan/data/Dacon_BRCA/test_segmented/


source /home/suhan/miniconda3/etc/profile.d/conda.sh
conda activate unet

while read f ;
do
python predict.py -i ${f} -m ./checkpoints/checkpoint_epoch5.pth -o /home/suhan/data/Dacon_BRCA/train_segmented/${f##*/}
done < /home/suhan/data/Dacon_BRCA/train_files.txt


while read f ;
do
python predict.py -i ${f} -m ./checkpoints/checkpoint_epoch5.pth -o /home/suhan/data/Dacon_BRCA/test_segmented/${f##*/}
done < /home/suhan/data/Dacon_BRCA/test_files.txt


conda deactivate
