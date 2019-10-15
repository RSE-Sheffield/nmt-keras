set -e

conf=config-dqTest-sentQEbRNN.py
task_name=wmt15
model_type=EncSent
model_name=${task_name}_srcmt_${model_type}
store_path=test_models/${model_name}/
patience=5

if [ $KERAS_BACKEND = "theano" ] ; then
  TESTVAL=0.1910756629756321 #PCC test value for sentence level, theano, BiRNN
elif [ $KERAS_BACKEND = "tensorflow" ] ; then
  TESTVAL=0.16354782921305733 #PCC test value for sentence level, theano, BiRNN
else
  echo "Keras backend environment variable incorrectly set"
  exit 1
fi

python utils/getTestData.py

PYTHONHASHSEED=0 python main.py || true > log-${model_name}-test.txt

PCC=$(awk -F, 'NR==1 {next};$3>M {M=$3};END {print M}' trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics)

echo "PCC test value was $PCC tested against $TESTVAL"

if echo $PCC $TESTVAL | awk '{exit ($1-$2)^2>1E-12}'; then
  echo "QE test passed (sentence level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
else
  echo "QE test failed (sentence level BiRNN with $KERAS_BACKEND on $task_name test dataset)"
fi
