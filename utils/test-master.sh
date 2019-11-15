#!/bin/sh

set -e

if [ "$TEST_MODE" = "sentence" ]; then
  ./utils/test-BiRNNsent.sh
elif [ "$TEST_MODE" = "document" ]; then
  ./utils/test-BiRNNdoc.sh
elif [ "$TEST_MODE" = "word" ]; then
  ./utils/test-BiRNNword.sh
else
  exit 1
fi
