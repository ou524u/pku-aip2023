import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


from myTensor import Tensor_Float as myTensor_f
from myTensor import Tensor_Int as myTensor_i
from operators import *

# Step 1: Data Preprocessing

def parse_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Convert PyTorch DataLoader to myTensor format
    def convert_to_my_tensor(data_loader):
        my_tensors = []
        real_labels = []
        for images, labels in data_loader:
            # 'images' is a batch of images
            # 'labels' is a batch of corresponding labels            
            my_tensor = myTensor_f(images.shape, "GPU", images.view(-1).tolist())  # Adjust based on your myTensor class
            my_label = myTensor_i(labels.shape,"GPU", labels.view(-1).tolist())
            my_tensors.append(my_tensor)
            real_labels.append(my_label)
        return my_tensors, real_labels

    # Convert train and test loaders to myTensor format
    X_tr, y_tr = convert_to_my_tensor(train_loader)
    X_te, y_te = convert_to_my_tensor(test_loader)

    return X_tr, y_tr, X_te, y_te


def list2Tensor(list_T):
    # from list of myTensor_f to a single Tensor

    shape = list_T[0].shape()[1:]
    all_shape = [len(list_T)] + shape
    all_data = []
    for l_tensor in list_T:
        all_data += l_tensor.data()

    print(f"@list2Tensor reshaped into {all_shape}")
    if isinstance(list_T[0], myTensor_f):
        return Tensor(myTensor_f(all_shape, "GPU", all_data))
    elif isinstance(list_T[0], myTensor_i):
        return Tensor(myTensor_i(all_shape, "GPU", all_data))
    raise ValueError("unknown myTensor typedef")
    


# Step 2: define the CNN
class SimpleCNN():
    def __init__(self):
        self.conv1 = Tensor(myTensor_f([32,1,3,3], "GPU").random(0,1))
        self.conv2 = Tensor(myTensor_f([64,32,3,3], "GPU").random(0,1))
        self.fcw1   = Tensor(myTensor_f([64*7*7, 128],"GPU").random(0,1))
        self.fcw2   = Tensor(myTensor_f([128, 10],"GPU").random(0,1))

        self.fcb1   = Tensor(myTensor_f([128],"GPU").random(0,1))
        self.fcb2   = Tensor(myTensor_f([10],"GPU").random(0,1))

        self.weights = [self.conv1, self.conv2, self.fcw1, self.fcw2, self.fcb1, self.fcb2]


    def forward(self, x, real_labels):
        x = maxpool(relu(conv(x, self.conv1)), kernel_shape=[2,2], stride_h=2, stride_w=2, pad_h=0, pad_w=0)
        x = maxpool(relu(conv(x, self.conv2)), kernel_shape=[2,2], stride_h=2, stride_w=2, pad_h=0, pad_w=0)
        x = flat(x)
        x = fc(fc(x, self.fcw1, self.fcb1), self.fcw2, self.fcb2)
        x, error = sftcrossen(x, real_labels)
        return x, error
    

    def SGD_epoch(self, X, y, lr=0.1, batch=100):
        num_examples = len(X)
        for start in range(0, num_examples, batch):
            end = start + batch
        # here X_batch and y_batch are a list of Tensors. we need to put them together so that
        # they should be shaped as [N, original_shape]
            X_batch = list2Tensor(X[start:end])
            y_batch = list2Tensor(y[start:end])

            # print(f"X_batch type is {X_batch.realize_cached_data().shape()}")
            
            tr_loss, tr_error = self.forward(X_batch, y_batch)
            tr_loss.backward()

            # update grad
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - lr * self.weights[i].grad

            return tr_loss, tr_error
        


    def Adam_epoch(self, X, y, lr=0.1, batch=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        t = 0
        m = [myTensor_f(weight.realize_cached_data().shape(), "GPU").zeros() for weight in self.weights]
        v = [myTensor_f(weight.realize_cached_data().shape(), "GPU").zeros() for weight in self.weights]

        num_examples = len(X)


        # Iterate over minibatches
        for start in range(0, num_examples, batch):
            t += 1
            end = start + batch

            # Forward pass
            X_batch = list2Tensor(X[start:end])
            y_batch = list2Tensor(y[start:end])

            tr_loss, tr_error = self.forward(X_batch, y_batch)
            tr_loss.backward()

            # Update first-order momentum vector
            for i in range(len(self.weights)):
                m[i] = beta1 * m[i] + (1 - beta1) * self.weights[i].grad

                # Update second-order momentum vector
                v[i] = beta2 * v[i] + (1 - beta2) * (self.weights[i].grad * self.weights[i].grad)

            # Bias correction
            m_hat = [m[i] / (1 - beta1**t) for i in range(len(m))]
            v_hat = [v[i] / (1 - beta2**t) for i in range(len(v))]

            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] -  lr * m_hat[i] / (sqrt(v_hat[i]) + epsilon)

            return tr_loss, tr_error
        


    def train_nn(self, X_tr, y_tr, X_te, y_te, 
                 epochs=10, lr=0.5, batch=100, 
                 beta1=0.9, beta2=0.999, use_Adam=False):
        
        X_testbatch = list2Tensor(X_te)
        y_testbatch = list2Tensor(y_te)
    
        print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
        for epoch in range(epochs):
            if not use_Adam:
                tr_loss, tr_error = self.SGD_epoch(X_tr, y_tr, lr=lr, batch=batch)
                te_loss, te_error = self.forward(X_testbatch, y_testbatch)
                print(f"| {epoch} | {tr_loss.realize_cached_data().data()} | {tr_error} | {te_loss.realize_cached_data().data()} | {te_error} |")

            else :
                tr_loss, tr_error = self.Adam_epoch(X_tr, y_tr, lr=lr, batch=batch)
                te_loss, te_error = self.forward(X_testbatch, y_testbatch)
                print(f"| {epoch} | {tr_loss.realize_cached_data().data()} | {tr_error} | {te_loss.realize_cached_data().data()} | {te_error} |")

        

# Step 3: train and test
                
model = SimpleCNN()
X_tr, y_tr, X_te, y_te = parse_mnist() 
epochs = 20
use_Adam = False

model.train_nn(X_tr, y_tr, X_te, y_te, epochs=epochs, 
               lr = 0.01, batch=100, beta1=0.9, beta2=0.999, 
               use_Adam=use_Adam)


