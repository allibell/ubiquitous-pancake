# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Section: Federated Learning

# # Lesson: Introducing Federated Learning
#
# Federated Learning is a technique for training Deep Learning models on data to which you do not have access. Basically:
#
# Federated Learning: Instead of bringing all the data to one machine and training a model, we bring the model to the data, train it locally, and merely upload "model updates" to a central server.
#
# Use Cases:
#
#     - app company (Texting prediction app)
#     - predictive maintenance (automobiles / industrial engines)
#     - wearable medical devices
#     - ad blockers / autotomplete in browsers (Firefox/Brave)
#     
# Challenge Description: data is distributed amongst sources but we cannot aggregated it because of:
#
#     - privacy concerns: legal, user discomfort, competitive dynamics
#     - engineering: the bandwidth/storage requirements of aggregating the larger dataset

# # Lesson: Introducing / Installing PySyft
#
# In order to perform Federated Learning, we need to be able to use Deep Learning techniques on remote machines. This will require a new set of tools. Specifically, we will use an extensin of PyTorch called PySyft.
#
# ### Install PySyft
#
# The easiest way to install the required libraries is with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/overview.html). Create a new environment, then install the dependencies in that environment. In your terminal:
#
# ```bash
# conda create -n pysyft python=3
# conda activate pysyft # some older version of conda require "source activate pysyft" instead.
# conda install jupyter notebook
# pip install syft
# pip install numpy
# ```
#
# If you have any errors relating to zstd - run the following (if everything above installed fine then skip this step):
#
# ```
# pip install --upgrade --force-reinstall zstd
# ```
#
# and then retry installing syft (pip install syft).
#
# If you are using Windows, I suggest installing [Anaconda and using the Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) to work from the command line. 
#
# With this environment activated and in the repo directory, launch Jupyter Notebook:
#
# ```bash
# jupyter notebook
# ```
#
# and re-open this notebook on the new Jupyter server.
#
# If any part of this doesn't work for you (or any of the tests fail) - first check the [README](https://github.com/OpenMined/PySyft.git) for installation help and then open a Github Issue or ping the #beginner channel in our slack! [slack.openmined.org](http://slack.openmined.org/)

import torch as th

x = th.tensor([1,2,3,4,5])
x

y = x + x

print(y)



import syft as sy

hook = sy.TorchHook(th)

th.tensor([1,2,3,4,5])

# # Lesson: Basic Remote Execution in PySyft

# ## PySyft => Remote PyTorch
#
# The essence of Federated Learning is the ability to train models in parallel on a wide number of machines. Thus, we need the ability to tell remote machines to execute the operations required for Deep Learning.
#
# Thus, instead of using Torch tensors - we're now going to work with **pointers** to tensors. Let me show you what I mean. First, let's create a "pretend" machine owned by a "pretend" person - we'll call him Bob.

bob = sy.VirtualWorker(hook, id="bob")

bob._objects

x = th.tensor([1,2,3,4,5])

x = x.send(bob)

bob._objects

x.location

x.id_at_location

x.id

x.owner

hook.local_worker

x

x = x.get()
x

bob._objects

# # Project: Playing with Remote Tensors
#
# In this project, I want you to .send() and .get() a tensor to TWO workers by calling .send(bob,alice). This will first require the creation of another VirtualWorker called alice.

alice = sy.VirtualWorker(hook, id="alice")

alice

x = th.tensor([8, 6, 7, 5, 3, 0, 9])

x.send(alice, bob)











# # Lesson: Introducing Remote Arithmetic

x = th.tensor([1,2,3,4,5]).send(bob)
y = th.tensor([1,1,1,1,1]).send(bob)

x

y

z = x + y

z

z = z.get()
z

z = th.add(x,y)
z

z = z.get()
z

x = th.tensor([1.,2,3,4,5], requires_grad=True).send(bob)
y = th.tensor([1.,1,1,1,1], requires_grad=True).send(bob)

z = (x + y).sum()

z.backward()

x = x.get()

x

x.grad



# # Project: Learn a Simple Linear Model
#
# In this project, I'd like for you to create a simple linear model which will solve for the following dataset below. You should use only Variables and .backward() to do so (no optimizers or nn.Modules). Furthermore, you must do so with both the data and the model being located on Bob's machine.

features_ptr = th.randn(64, 128, requires_grad=True).send(bob)
targets_ptr = th.randn(64, 1, requires_grad=True).send(bob)
w1_ptr = th.randn(128, 1, requires_grad=True).send(bob)
b1_ptr = th.randn(1,1).send(bob)

for i in range(10):
    pred_ptr = th.mm(features_ptr, w1_ptr) + b1_ptr
    loss_ptr = ((pred_ptr - targets_ptr) ** 2).sum()
    loss_ptr.backward()

    learning_rate = 0.1
    w1_ptr.data.sub_(w1_ptr.grad * learning_rate)
    w1_ptr.grad *= 0

    print(loss_ptr.get().data)

# # Lesson: Garbage Collection and Common Errors
#

bob = bob.clear_objects()

bob._objects

x = th.tensor([1,2,3,4,5]).send(bob)

bob._objects

del x

bob._objects

x = th.tensor([1,2,3,4,5]).send(bob)

bob._objects

x = "asdf"

bob._objects

x = th.tensor([1,2,3,4,5]).send(bob)

x

bob._objects

x = "asdf"

bob._objects

del x

bob._objects

bob = bob.clear_objects()
bob._objects

for i in range(1000):
    x = th.tensor([1,2,3,4,5]).send(bob)

bob._objects

x = th.tensor([1,2,3,4,5]).send(bob)
y = th.tensor([1,1,1,1,1])

z = x + y

x = th.tensor([1,2,3,4,5]).send(bob)
y = th.tensor([1,1,1,1,1]).send(alice)

z = x + y



# # Lesson: Toy Federated Learning
#
# Let's start by training a toy model the centralized way. This is about a simple as models get. We first need:
#
# - a toy dataset
# - a model
# - some basic training logic for training a model to fit the data.

from torch import nn, optim

# A Toy Dataset
data = th.tensor([[1.,1],[0,1],[1,0],[0,0]], requires_grad=True)
target = th.tensor([[1.],[1], [0], [0]], requires_grad=True)

# A Toy Model
model = nn.Linear(2,1)

opt = optim.SGD(params=model.parameters(), lr=0.1)


# +
def train(iterations=20):
    for iter in range(iterations):
        opt.zero_grad()

        pred = model(data)

        loss = ((pred - target)**2).sum()

        loss.backward()

        opt.step()

        print(loss.data)
        
train()
# -

data_bob = data[0:2].send(bob)
target_bob = target[0:2].send(bob)

data_alice = data[2:4].send(alice)
target_alice = target[2:4].send(alice)

datasets = [(data_bob, target_bob), (data_alice, target_alice)]


def train(iterations=20):

    model = nn.Linear(2,1)
    opt = optim.SGD(params=model.parameters(), lr=0.1)
    
    for iter in range(iterations):

        for _data, _target in datasets:

            # send model to the data
            model = model.send(_data.location)

            # do normal training
            opt.zero_grad()
            pred = model(_data)
            loss = ((pred - _target)**2).sum()
            loss.backward()
            opt.step()

            # get smarter model back
            model = model.get()

            print(loss.get())


train()



# # Lesson: Advanced Remote Execution Tools
#
# In the last section we trained a toy model using Federated Learning. We did this by calling .send() and .get() on our model, sending it to the location of training data, updating it, and then bringing it back. However, at the end of the example we realized that we needed to go a bit further to protect people privacy. Namely, we want to average the gradients BEFORE calling .get(). That way, we won't ever see anyone's exact gradient (thus better protecting their privacy!!!)
#
# But, in order to do this, we need a few more pieces:
#
# - use a pointer to send a Tensor directly to another worker
#
# And in addition, while we're here, we're going to learn about a few more advanced tensor operations as well which will help us both with this example and a few in the future!

bob.clear_objects()
alice.clear_objects()



x = th.tensor([1,2,3,4,5]).send(bob)

x = x.send(alice)

bob._objects

alice._objects

y = x + x

y

bob._objects

alice._objects

jon = sy.VirtualWorker(hook, id="jon")

# +
bob.clear_objects()
alice.clear_objects()

x = th.tensor([1,2,3,4,5]).send(bob).send(alice)
# -

bob._objects

alice._objects

x = x.get()
x

bob._objects

alice._objects

x = x.get()
x

bob._objects

# +
bob.clear_objects()
alice.clear_objects()

x = th.tensor([1,2,3,4,5]).send(bob).send(alice)
# -

bob._objects

alice._objects

del x

bob._objects

alice._objects





# # Lesson: Pointer Chain Operations

bob.clear_objects()
alice.clear_objects()

x = th.tensor([1,2,3,4,5]).send(bob)

bob._objects

alice._objects

x.move(alice)

bob._objects

alice._objects



x = th.tensor([1,2,3,4,5]).send(bob).send(alice)

bob._objects

alice._objects

x.remote_get()

bob._objects

alice._objects

x.move(bob)

x

bob._objects

alice._objects














# -
# # Section Project:
#
# For the final project for this section, you're going to train on the MNIST dataset using federated learning However the gradient should not come up to central server in raw form

import torch
from torchvision import transforms
import torchvision.datasets as torch_datasets

transform = transforms.Compose([transforms.ToTensor()])

mnist_trainset = torch_datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = torch_datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

# +
import syft as sy

hook = sy.TorchHook(torch)


# +
from torch import nn, optim
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # flatten
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_sfotmax(self.fc4(x), dim-1)

        return x

model = MNISTClassifier()
criterion = nn.NLLLoss()
otpimizer = optim.Adam(model.parameters(), lr=0.003)


# -

def create_workers(trainloader, testloader, num_workers):
    workers = [sy.VirtualWorker(hook, id=f"worker_{i}") for i in range(num_workers)]

    train_batches_per_worker = len(trainloader) // len(workers)
    worker_train_idxs = range(0, len(trainloader), train_batches_per_worker)
    worker_train_idxs[-1] += len(trainloader) % len(workers)
    worker_train_data = [
                            trainloader[idx:idx+1].send(worker)
                            for idx, worker in zip(worker_train_idxs, workers)
                        ]

    test_batches_per_worker = len(testloader) // len(workers)
    worker_test_idxs = range(0, len(testloader), test_batches_per_worker)
    worker_test_idxs[-1] += len(testloader) % len(workers)
    worker_test_data = [
                            testloader[idx:idx+1].send(worker)
                            for idx, worker in zip(worker_test_idxs, workers)
                       ]

    datasets = list(zip(worker_train_data, worker_test_data))

    return workers, datasets

def train_federated(trainloader, testloader, num_workers=24, epochs=30):
    train_losses, test_losses = [], []

    workers, datasets = create_workers(trainloader, testloader, num_workers)


datasets


