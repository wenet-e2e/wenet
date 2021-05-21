import logging
import os

# worker config
workers = 10
worker_class = 'gevent'
worker_connections = 1000
threads = 2

bind = '0.0.0.0:15000'
daemon = False

keepalive = 2
timeout = 60
max_requests = 1024
backlog = 65535

chdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

