import urllib.request
import os

os.makedirs("data", exist_ok=True)

url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"

urllib.request.urlretrieve(url, "mnist.pkl.gz")

print("MNIST downloaded")