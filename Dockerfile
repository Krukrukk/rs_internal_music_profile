FROM python:3.9

WORKDIR /usr/src/app

# Install necessary data
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

# start fastapi server. Might be helfull
CMD ["python", "server.py"]
