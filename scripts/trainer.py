import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_set import CustomDataset
from utils.to_tensor import ToTensor
from utils.rescale import Rescale
from utils.random_crop import RandomCrop

from models.model import CustomModel


def train_net(neural_net, n_epochs, criterion, train_loader, optimizer):

    # prepare the net for training
    neural_net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            steering_throttle = data['steering_throttle']

            # flatten pts
            steering_throttle = steering_throttle.view(steering_throttle.size(0), -1)

            # convert variables to floats for regression loss
            steering_throttle = steering_throttle.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = neural_net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, steering_throttle)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    net = CustomModel()
    n_epoch = 1
    batch_size = 8
    optimizer = optim.Adam(params=net.parameters(), lr=0.1)

    data_transform = transforms.Compose([ToTensor()])
    transformed_dataset = CustomDataset(csv_file='../data/data.csv', root_dir='../data/', transform=data_transform)
    data_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_net(net, n_epoch, nn.MSELoss(), data_loader, optimizer)
    torch.save(net.state_dict(), "./saved_models/my_model.pt")
