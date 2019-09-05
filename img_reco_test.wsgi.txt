#! /usr/bin/python

import sys
sys.path.insert(0,"/var/www/img_reco_test")
sys.path.insert(0,"/opt/conda/lib/python3.6/site-packages")
sys.path.insert(0,"/opt/conda/bin/")

import os
os.environ['PYTHONPAYH'] = 'opt/conda/bin/python'

from img_reco_test import app as application