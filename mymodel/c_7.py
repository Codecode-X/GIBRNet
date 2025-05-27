import torch
import torch.nn
import torch.autograd

class c_7(torch.nn.Module):
    def __init__(self):
        super(c_7, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(2, 2))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.linear = torch.nn.Linear(in_features=36, out_features=6)


        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.weight_fc = torch.nn.Linear(in_features=24, out_features=1)

        self.frame_messagepass_fc = torch.nn.Linear(in_features=6, out_features=6)
        self.last_messagepass_fc = torch.nn.Linear(in_features=6, out_features=6)

        self.output_fc1 = torch.nn.Linear(in_features=240, out_features=120)
        self.output_fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.output_fc3 = torch.nn.Linear(in_features=60, out_features=6)


    def node_updata(self, frame_id, pos, message):   #[5,6] [4,6] 
        if frame_id == 0: 
            new_pos = torch.sum((message), 0)
        else: 
            new_pos = torch.sum((message), 0) + self.last_messagepass_fc(pos[frame_id-1, :])
        return new_pos   #[6]

    def aggregate_weight(self,edge_feature):   #[4,24]
        weight = self.weight_fc(edge_feature).squeeze()   #[4]
        weight = self.sigmoid(weight)   #[4]
        return weight   #[4]


    def forward(self, nodes, pos, attmap, depths):
        N = nodes.shape[0]
        nodes_feature = self.conv1(nodes.view(N*5*4, 3, 224, 224))   #[20N,3,111,111]
        nodes_feature = self.pool1(nodes_feature)   #[20N,3,37,37]
        nodes_feature = self.relu(nodes_feature)
        nodes_feature = self.conv2(nodes_feature)
        nodes_feature = self.pool2(nodes_feature)
        nodes_feature = self.relu(nodes_feature)
        nodes_feature = self.linear(nodes_feature.view(N, 5, 4, -1))   #[N,5,4,6]


        pos_updata = torch.autograd.Variable(torch.zeros(size=(N,5,4,6))).cuda()   #[N,5,4,6]

        for b_id in range(N):  
            for frame_id in range(5): 
                message_aggregate_weight = attmap[b_id,frame_id,...]   #[4,4]
                for n_id in range(4):  
                    pos_message = self.frame_messagepass_fc(pos[b_id, frame_id,...])   #[4,6]  
                    message_aggregate = message_aggregate_weight[:, n_id].view(4,1).expand_as(pos_message) * pos_message   #[4,6]  
                    #                             [4]                                   [4]                          [4,6]
                    pos_updata[b_id, frame_id, n_id, :] = self.node_updata(frame_id, pos[b_id, :, n_id, :], message_aggregate)

        input_feature = torch.cat((nodes_feature, pos_updata), -1).view(N,-1)   #[N,5,4,12]->[N,240]
        result = self.output_fc1(input_feature)   #[N,120]
        result = self.relu(result)
        result = self.output_fc2(result)   #[N,60]
        result = self.relu(result)
        result = self.output_fc3(result)   #[N,6]
        return result