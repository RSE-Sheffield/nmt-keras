set -e

conf=config-dqTest-sentQEbRNN.py
task_name=wmt15
model_type=EncSent
model_name=${task_name}_srcmt_${model_type}
store_path=test_models/${model_name}/
patience=5
TESTVAL=0.1910756629756321 #PCC test value for sentence level, theano, BiRNN

python utils/getTestData.py

PYTHONHASHSEED=0 python main.py || true > log-${model_name}-test.txt

PCC=$(awk -F, 'NR==1 {next};$3>M {M=$3};END {print M}' trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics)

if [ "$PCC" = "$TESTVAL" ]; then
  echo "QE test passed (sentence level BiRNN with Theano on $task_name test dataset)"
  echo "PCC test value was $PCC which is equal to $TESTVAL"
  exit 0
fi
