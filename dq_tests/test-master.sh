#!/bin/sh

set -e

if [ "$TEST_MODE" = "sentence" ]; then
  ./dq_tests/test-BiRNNsent.sh
elif [ "$TEST_MODE" = "document" ]; then
  ./dq_tests/test-BiRNNdoc.sh
elif [ "$TEST_MODE" = "word" ]; then
  ./dq_tests/test-BiRNNword.sh
else
  exit 1
fi
