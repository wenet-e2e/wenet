from flask import Flask
from hashlib import sha1
import os
import time

def create_secret():
    return sha1(bytes('%s%s' % (
        os.urandom(16), time.time()),
        encoding='utf-8')).hexdigest()

server = Flask(__name__)
random_secret = create_secret()
server.secret_key = random_secret

# api
# flake8: noqa
from api.asr import *
