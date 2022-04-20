#!/bin/bash

# sudo apt install rename
# apt-get install rename

cd /opt/ml/input/data/ICDAR17_raw/ch8_training_gt
rename -v 's/img/tr_img/' *.txt

cd /opt/ml/input/data/ICDAR17_raw/ch8_training_images
rename -v 's/img/tr_img/' *.jpg
rename -v 's/img/tr_img/' *.png
rename -v 's/img/tr_img/' *.gif

cd /opt/ml/input/data/ICDAR17_raw/ch8_validation_gt
rename -v 's/img/va_img/' *.txt

cd /opt/ml/input/data/ICDAR17_raw/ch8_validation_images
rename -v 's/img/va_img/' *.jpg
rename -v 's/img/va_img/' *.png
rename -v 's/img/va_img/' *.gif