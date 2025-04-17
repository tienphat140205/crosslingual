# /bin/bash
set -e


python crosslingual_topic_model.py --model DualTopic --dataset ECNews --weight_MI 30.0

python crosslingual_topic_model.py --model DualTopic --dataset AmazonReview --weight_MI 50.0

python crosslingual_topic_model.py --model DualTopic --dataset RakutenAmazon --weight_MI 50.0
