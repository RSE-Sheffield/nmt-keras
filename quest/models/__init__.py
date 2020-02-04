#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# __init__.py
#
# Copyright (C) 2019 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#



## LIST HERE THE QE MODELS THAT ARE AVAILABLE
## /!\ keep the name of the model in lowercase

QE_MODELS = {
'encword':          'birnn_word',
'encsent':          'birnn_sent',
'encdoc':           'birnn_doc',
'encdocatt':        'birnn_doc_att',
'predictor':        'predictor',
'estimatorword':    'estimator_word',
'estimatorsent':    'estimator_sent',
'estimatordoc':     'estimator_doc',
}


import importlib

# QE MODEL FACTORY-like
def get(model_name, params):
    try:
        qe_model = getattr(importlib.import_module('quest.models.{}'.format(QE_MODELS[model_name.lower()])), model_name)
        return qe_model(params)

    except ValueError as e:
        print("/!\ {}".format(e))
