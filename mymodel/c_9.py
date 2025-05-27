import torch
import torch.nn
import torch.autograd

class c_9(torch.nn.Module):
    def __init__(self):
        super(c_9, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(2, 2))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.linear = torch.nn.Linear(in_features=36, out_features=6)

        self.depth_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2))
        self.depth_pool1 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.depth_conv2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(2, 2))
        self.depth_pool2 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.depth_linear = torch.nn.Linear(in_features=36, out_features=6)

        self.relu = torch.nn.ReLU()

        self.output_fc1 = torch.nn.Linear(in_features=360, out_features=180)
        self.output_fc2 = torch.nn.Linear(in_features=180, out_features=60)
        self.output_fc3 = torch.nn.Linear(in_features=60, out_features=6)



    def forward(self, nodes, pos, attmap, depths):

        N = nodes.shape[0]
        nodes_feature = self.conv1(nodes.view(N*5*4, 3, 224, 224))   #[20N,3,111,111]
        nodes_feature = self.pool1(nodes_feature)   #[20N,3,37,37]
        nodes_feature = self.relu(nodes_feature)
        nodes_feature = self.conv2(nodes_feature)
        nodes_feature = self.pool2(nodes_feature)
        nodes_feature = self.relu(nodes_feature)
        nodes_feature = self.linear(nodes_feature.view(N, 5, 4, -1))   #[N,5,4,6]
        nodes_depth_feature = self.depth_conv1(depths.view(N * 5 * 4, 3, 224, 224))  # [20N,3,111,111]
        nodes_depth_feature = self.depth_pool1(nodes_depth_feature)  # [20N,3,37,37]
        nodes_depth_feature = self.relu(nodes_depth_feature)
        nodes_depth_feature = self.depth_conv2(nodes_depth_feature)
        nodes_depth_feature = self.depth_pool2(nodes_depth_feature)
        nodes_depth_feature = self.relu(nodes_depth_feature)
        nodes_depth_feature = self.depth_linear(nodes_depth_feature.view(N, 5, 4, -1))  # [N,5,4,6]

        input_feature = torch.cat((nodes_feature, nodes_depth_feature, pos), -1).view(N,-1)   #[N,360]
        result = self.output_fc1(input_feature)   #[N,180]
        result = self.relu(result)
        result = self.output_fc2(result)   #[N,60]
        result = self.relu(result)
        result = self.output_fc3(result)   #[N,6]
        return result