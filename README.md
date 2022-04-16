# Sequential Non-rigid Quantization algorithm (SNRQ)
Python implementation of "Beyond neighbourhood-preserving transformations for quantization-based unsupervised hashing" paper.

# How to use?
1- Download and unzip the dataset: <br />
[mnist_gist512](https://www.dropbox.com/s/97suefbk4aaa26c/mnist_gist512.zip?dl=0) <br />
[labelme_vggfc7](https://www.dropbox.com/s/0nc80qepzj8615f/labelme_vggfc7.rar?dl=0) <br />
[cifar10_vggfc7](https://www.dropbox.com/s/bnybq48ljtsyuit/cifar10_vggfc7.rar?dl=0) <br />
[nuswide_vgg](https://www.dropbox.com/s/6hl9t6oy78w028d/nuswide_vgg.rar?dl=0) <br />
[colorectal_EfficientNet](https://www.dropbox.com/s/wdsalhu73bnrtsg/colorectal_EfficientNet.rar?dl=0) <br />

2- Complete the parameter initialization in demo_SNRQ.py
For example:

method_name='SNRQ'
path = 'MNIST' # path to folder containing dataset
dataset_name = 'mnist_gist512' #options: mnist_gist512, cifar10_vggfc7,
labelme_vggfc7, nuswide_vgg, colorectal_efficientnet
K = 16 # number of bits
alpha=3 # Control quantization power
beta=.01 # Control (Non-rigidness) the trade-off between corrupting neighbourhood structure of data and of minimizing quantization loss.
analytic_derivatve=True #if False, the derivative is calculated numerically uing which is slower in L-BFGS-B optimizer.
n_iter=70 #number of iterations
3- Run the demo_SNRQ.py

# If you need further training speed, implement the derivative using Tensorflow and use GPU
