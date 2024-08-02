import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
from testnn.CustomDataSet import CustomImageDataset

PATH = 'customdatahere' # change
LABEL_PATH = 'labels' # change
# Change both once 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

training_data = CustomImageDataset(
    annotations_file=PATH,
    img_dir=LABEL_PATH,
    transform=ToTensor()
)

test_data = CustomImageDataset(
    annotations_file=PATH,
    img_dir=LABEL_PATH,
    transform=ToTensor()
)
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #Converts each 2D 28x28 image into a contiguous array of 784 pixel values
        self.linear_relu_stack = nn.Sequential( #Ordered container of modules, passed through all the modules in the same order as defined
            nn.Linear(28*28, 512), #Module that applies a linear transformation ont he input using its stored weights and biases
            nn.ReLU(), #Non-linear activations create the complex mappings between the model's inputs and output, intorduce non linearity to help nns learn a wide variety of phenomena
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model= NeuralNetwork().to(device)

"""
Most used algorithm is backpropagation, where parameters are adjusted according to the gradient of the loss function with respect to the given parameter\
torch.autograd automaticallu computes gradients for any computational graph
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

learning_rate = 1e-3 #How much to update models parameteres at each batch/epoch
batch_size = 64 #Number of data samples propagated through the network before the parameters are updated
epochs = 10 #Number of times to iterate over dataset
PATH = "model.pt"
LOSS = 0.4

loss_fn = nn.CrossEntropyLoss() #measures degree of dissimilarity of obtained result to the target value, want to minimize

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #Adjusts model parameters to reduce model error in each training step

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.train()

for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

