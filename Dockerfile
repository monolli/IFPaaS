FROM python:3-slim
# Install requirements

COPY requirements.txt /
RUN pip install -r /requirements.txt

# Copy to a specific path
COPY main.py /app/
COPY iris /app/iris/
COPY model /app/model/

# Change the working directory
WORKDIR /app

# Expose de port 8000
EXPOSE 8000

# Set the webservice execution as entrypoint
ENTRYPOINT gunicorn --reload -w 3 -b :8000 main:app
