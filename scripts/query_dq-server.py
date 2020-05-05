#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# query_dq-server.py
#
# Copyright (C) 2020 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

"""
-- USAGE

# Start the server:
    `python run_dq_server.py -c [config]`

# Submita a request via Python:
    `python query_dq-server.py`

(Inspired by https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
"""

# import the necessary packages
import requests
import json

# initialize the Keras REST API endpoint URL along with the input
KERAS_REST_API_URL = "http://localhost:5000/predict"

# TOY EXAMPLE
payload = {
"tgtScore": "-1.7567363",
"tgt": "This is a test.",
"tgtTokens": ["Ahoj", "světe", "!"],
"tgtScores": ["-1.7609119415283203", "-11.347211837768555", "-10.362051010131836"],
"src": "C'est un test.",
"srcTokens": ["Hello", "World", "!"],
"alignment": [[0,0], [1,1], [2,2],]
}

# submit the request
r = requests.post(KERAS_REST_API_URL, json=json.dumps(payload)).json()

# ensure the request was sucessful
if r["success"]:
    # loop over the predictions and display them
        if r['predlevel'] == 'word':
            for segid in r['predictions'].keys():
                # print("SegID {}: {}".format(segid, ' '.join(r['predictions'][segid])))
                print("{}".format(' '.join(r['predictions'][segid])))
        else:
            pred = ["{:.4f}".format(pred) for pred in r['predictions']]
            print(','.join(pred))

# otherwise, the request failed
else:
    print("Request failed")
