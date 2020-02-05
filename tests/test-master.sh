#!/bin/sh

set -e

if [ "$TEST_MODE" = "sentence" ]; then
  ./tests/test-BiRNNsent.sh
elif [ "$TEST_MODE" = "document" ]; then
  ./tests/test-BiRNNdoc.sh
elif [ "$TEST_MODE" = "word" ]; then
  ./tests/test-BiRNNword.sh
else
  exit 1
fi
