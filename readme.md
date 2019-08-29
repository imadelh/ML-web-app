# Train and Deploy Machine Learning Model With Web Interface - Docker, PyTorch & Flask

Live access (deployed on GCP): https://ml-app.imadelhanafi.com

![alt text](https://imadelhanafi.com/data/draft/capture_app_elhanafi.gif)

---

Blog post: https://imadelhanafi.com/posts/train_deploy_ml_model/

This repo contains code associated with the above blog post. 



## Running on Local/cloud machine

Clone the repo and build the docker image

```
sudo docker build -t flaskml .
```

NB: if you have MemoryError while installing PyTorch in the container, please consider adding 2G swap to your virtual machine (https://linuxize.com/post/how-to-add-swap-space-on-ubuntu-18-04/)

Then after that you can run the container while specefying the absolute path to the app 

```
sudo docker run -i -t --rm -p 8888:8888 -v **absolute path to app directory**:/app flaskml
```

This will run the application on localhost:8888

You can use serveo.net or Ngrok to port the application to the web.

## Running on Jetson-Nano 

On Jetson-nano, to avoid long running time to build the image, you can download it from Docker Hub. 
We will also use a costumized Docker command https://gist.github.com/imadelh/cf7b12c9cc81c3cb95ad2c6bc747ccd0 to be able to access the GPU of the device on the container.

```
docker pull imadelh/jetson_pytorch_flask:arm_v1
```

Then on your device you can access the bash (this the default command on that image) 

```
sudo ./mydocker.sh run -i -t --rm -v /home/imad:/home/root/ imadelh/jetson_pytorch_flask:arm_v1

```

and then simply get to the application directory and run it

```
cd app
python3 app.py
```

## Useful files 

- Training and saving the CNN model : https://gist.github.com/imadelh/b337c7b16899831d80d9221a9a60e09f
- Visualize the inference : https://colab.research.google.com/github/imadelh/ML-web-app/blob/master/Notebooks/emnist_inference_cnn-2.ipynb


## Info

This a generic web app for ML models. You can update your the network and weights by changing the following files. 

```
app/ml_model/network.py
app/ml_model/trained_weights.pth
```


---
Imad El Hanafi
