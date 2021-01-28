FROM imadelh/jetson_pytorch_flask:ainize
EXPOSE 8888

ADD . /app
WORKDIR /app

# This is the runtime command for the container
CMD python3 app.py
