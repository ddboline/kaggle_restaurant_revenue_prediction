#!/bin/bash

scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_restaurant_revenue_prediction/restaurant_revenue_prediction.tar.gz .
tar zxvf restaurant_revenue_prediction.tar.gz
rm restaurant_revenue_prediction.tar.gz

./my_model.py $1 > output_${1}.out 2> output_${1}.err

D=`date +%Y%m%d%H%M%S`
tar zcvf output_${1}_${D}.tar.gz model_${1}.pkl.gz output_${1}.out output_${1}.err
scp output_${1}_${D}.tar.gz ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_restaurant_revenue_prediction/
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk done_kaggle_restaurant_revenue_prediction_${1}"
echo "JOB DONE ${1}"
