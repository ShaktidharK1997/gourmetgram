#!/bin/bash
# Create the log file to be able to run tail
touch /var/log/cron.log

# Start cron
cron

# Follow the logs
tail -f /var/log/cron.log