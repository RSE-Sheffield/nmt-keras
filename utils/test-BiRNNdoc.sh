#!/bin/sh

set -e

conf=config-dqTest-docQEbRNN.py
conf_predict=config-dqTest-docQEbRNNEval.py
task_name=testData-doc
model_type=EncSent
model_name=${task_name}_srcmt_${model_type}
store_path=trained_models/${model_name}/
patience=5
level=document
metric=pearson

# look in utils/testVals.csv for the correct value to compare deepquest test run against
TESTVAL=$(awk -v backend=$KERAS_BACKEND -v level=$level -v task_name=$task_name -v metric=$metric -F,\
 'NR==1 {next};$1==backend && $2==level && $3==task_name && $4==metric {M=$5};END {print M}' utils/testVals.csv )

rm -rf config.*
ln -s utils/$conf ./config.py

python utils/getTestData_BiRNNdoc.py

PYTHONHASHSEED=0 python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} SRC_LAN=src TRG_LAN=mt MODEL_TYPE=$model_type MODEL_NAME=$model_name PATIENCE=$patience SAVE_EACH_EVALUATION=True || true > log-${model_name}-test.txt

PCC=$(awk -F, '/pearson/ {M=-1;next};$3>M {M=$3};END {print M}' trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics)

echo "PCC test value was $PCC tested against $TESTVAL"

if echo $PCC $TESTVAL | awk '{exit ($1-$2)^2<1E-12}'; then
  TRAIN_RESULT="failed"
  echo "QE training $TRAIN_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
  exit 1
else
  TRAIN_RESULT="passed"
  echo "QE training $TRAIN_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
  rm -rf config.*
  ln -s utils/$conf_predict ./config.py
  EPOCH=$(awk -F, '/pearson/ {M=-1;next};$3>M {M=$3;E=$1};END {print E}' trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics)
  PYTHONHASHSEED=0 python predict.py --dataset datasets/Dataset_${task_name}_srcmt.pkl --model trained_models/${model_name}/epoch_${EPOCH}.h5 --save_path saved_predictions/prediction_${task_name}/
  PCC=$(awk -F, '/pearson/ {M=-1;next};$3>M {M=$3};END {print M}' saved_predictions/prediction_${task_name}/test.qe_metrics)
  TESTVAL=$(awk -v backend=$KERAS_BACKEND -v level=${level}Predict -v task_name=$task_name -v metric=$metric -F,\
   'NR==1 {next};$1==backend && $2==level && $3==task_name && $4==metric {M=$5};END {print M}' utils/testVals.csv )
  echo "PCC test value was $PCC tested against $TESTVAL"
  if echo $PCC $TESTVAL | awk '{exit ($1-$2)^2<1E-10}'; then
     PREDICT_RESULT="failed"
     echo "QE prediction $PREDICT_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
     exit 1
   else
     PREDICT_RESULT="passed"
     echo "QE prediction $PREDICT_RESULT ($level  level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
     exit 0
   fi
fi