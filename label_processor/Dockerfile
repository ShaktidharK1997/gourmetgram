FROM python:3.9-slim

WORKDIR /app

# Install cron
RUN apt-get update && apt-get -y install cron

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy script and startup files
COPY process_labels.py .
COPY random_sampler.py /app/random_sampler.py
RUN chmod +x /app/random_sampler.py
COPY crontab /etc/cron.d/process-labels-cron
COPY start.sh .

# Give execution permissions
RUN chmod 0644 /etc/cron.d/process-labels-cron
RUN chmod +x /app/start.sh

# Apply cron job
RUN crontab /etc/cron.d/process-labels-cron

# Run the command on container startup
CMD ["/app/start.sh"]