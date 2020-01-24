#!/bin/sh

set -e

conf=dq_tests/config-word-BiRNN.yml
task_name=testData-word
model_type=EncWord
model_name=${task_name}_srcmt_${model_type}
store_path=trained_models/${model_name}/
level=word
metric=f1_prod

# look in dq_tests/testVals.csv for the correct value to compare deepquest test run against
TESTVAL=$(awk -v backend=$KERAS_BACKEND -v level=$level -v task_name=$task_name -v metric=$metric -F,\
 'NR==1 {next};$1==backend && $2==level && $3==task_name && $4==metric {M=$5};END {print M}' dq_tests/testVals.csv )

python dq_tests/getTestData_BiRNNword.py

python __main__.py train -c ${conf} TASK_NAME=$task_name DATASET_NAME=$task_name MODEL_TYPE=$model_type MODEL_NAME=$model_name || true > log-${model_name}-test.txt

PCC=$(awk -F, '/f1_prod/ {M=-1;next};$2>M {M=$2};END {print M}' trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics)

echo "PCC test value was $PCC tested against $TESTVAL"

if echo $PCC $TESTVAL | awk '{exit ($1-$2)^2<1E-12}'; then
  TRAIN_RESULT="failed"
  echo "QE training $TRAIN_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
  exit 1
else
  TRAIN_RESULT="passed"
  echo "QE training $TRAIN_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
  EPOCH=$(awk -F, '/f1_prod/ {M=-1;next};$2>M {M=$2;E=$1};END {print E}' trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics)
  python __main__.py predict --dataset ${store_path}/Dataset_${task_name}_srcmt.pkl --model trained_models/${model_name}/epoch_${EPOCH}.h5 --save_path saved_predictions/prediction_${task_name}/ --evalset test
  PCC=$(awk -F, '/f1_prod/ {M=-1;next};$2>M {M=$2};END {print M}' saved_predictions/prediction_${task_name}/test.qe_metrics)
  TESTVAL=$(awk -v backend=$KERAS_BACKEND -v level=${level}Predict -v task_name=$task_name -v metric=$metric -F,\
   'NR==1 {next};$1==backend && $2==level && $3==task_name && $4==metric {M=$5};END {print M}' dq_tests/testVals.csv )
  echo "PCC test value was $PCC tested against $TESTVAL"
  if echo $PCC $TESTVAL | awk '{exit ($1-$2)^2<1E-12}'; then
     PREDICT_RESULT="failed"
     echo "QE prediction $PREDICT_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
     exit 1
   else
     PREDICT_RESULT="passed"
     echo "QE prediction $PREDICT_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
     exit 0
   fi
fi
