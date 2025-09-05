
FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install AWS CLI using pip (no apt needed)
RUN pip install awscli --upgrade

CMD ["python3", "app.py"]
