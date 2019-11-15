#!/bin/sh

set -e

conf=config-dqTest-docQEbRNN.py
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

PYTHONHASHSEED=0 python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} STORE_PATH=${store_path} SRC_LAN=src TRG_LAN=mt PRED_SCORE=mqm OUT_ACTIVATION=sigmoid MODEL_TYPE=$model_type MODEL_NAME=$model_name NEW_EVAL_ON_SETS=val PATIENCE=$patience SAVE_EACH_EVALUATION=True || true > log-${model_name}-test.txt

PCC=$(awk -F, 'NR==1 {next};$3>M {M=$3};END {print M}' trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics)

echo "PCC test value was $PCC tested against $TESTVAL"

if echo $PCC $TESTVAL | awk '{exit ($1-$2)^2>1E-12}'; then
  echo "QE test passed ($level level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
  exit 0
else
  echo "QE test failed ($level level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
  exit 1
fi
