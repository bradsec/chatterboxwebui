#!/usr/bin/env python3
"""
Production server configuration for Chatterbox Web UI

This file provides production-ready settings and can be used to run
the application with gunicorn for better performance and stability.

Usage:
    gunicorn -c production.py server:app
"""

import os
import multiprocessing

# Bind configuration
bind = f"{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '5000')}"

# Worker configuration
workers = int(os.environ.get('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "eventlet"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = os.environ.get('LOG_LEVEL', 'info')
accesslog = '-'
errorlog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'chatterbox-webui'

# Security
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 8192

# Timeout settings
timeout = 300  # Extended for TTS generation
keepalive = 5
graceful_timeout = 30

# SSL (if certificates are available)
keyfile = os.environ.get('SSL_KEYFILE')
certfile = os.environ.get('SSL_CERTFILE')

# Preload application for better memory usage
preload_app = True

# Process management
pidfile = '/tmp/chatterbox-webui.pid'
daemon = False

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Chatterbox Web UI server is ready")

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info("Worker received SIGINT or SIGQUIT")

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Chatterbox Web UI server is shutting down")