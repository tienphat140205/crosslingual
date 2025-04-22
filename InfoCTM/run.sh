# /bin/bash
set -e

python crosslingual_topic_model.py --model InfoCTM --dataset Amazon_Review --weight_MI 50 --lambda_contrast 50 --infoncealpha 50 --device 1

cd /mnt/MinhNV/InfoCTM/CNPMI
python CNPMI.py \
  --topics1 /mnt/MinhNV/InfoCTM/output/Amazon_Review/InfoCTM_K50_T15_en.txt \
  --topics2 /mnt/MinhNV/InfoCTM/output/Amazon_Review/InfoCTM_K50_T15_cn.txt \
  --ref_corpus_config configs/ref_corpus/en_zh.yaml 

cd /mnt/MinhNV/InfoCTM/utils
python TU.py --data /mnt/MinhNV/InfoCTM/output/Amazon_Review/InfoCTM_K50_T15_en.txt
python TU.py --data /mnt/MinhNV/InfoCTM/output/Amazon_Review/InfoCTM_K50_T15_cn.txt
