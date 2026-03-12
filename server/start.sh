#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables for production
export FLASK_ENV=production
export PORT=${PORT:-5000}

# Run database migrations if needed
echo "Running database migrations..."
flask db upgrade

# Start the application with Gunicorn
echo "Starting application on port $PORT..."
exec gunicorn --config gunicorn.conf.py app:app
