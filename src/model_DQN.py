import torch
import torch.nn as nn
import torch.nn.functional as F
from .config_DQN import gamma, sequence_length, device
# torch.manual_seed(0)

class QNet(nn.Module):
    def __init__(self, num_grid,num_feature, num_outputs):
        super(QNet, self).__init__()
        self.num_grid = num_grid
        self.num_feature = num_feature
        if self.num_grid == 48:
            self.num_out_conv = 2
        elif self.num_grid == 64:
            self.num_out_conv = 4
        elif self.num_grid == 128:
            self.num_out_conv = 12
        elif self.num_grid == 192:
            self.num_out_conv = 20
        elif self.num_grid == 256:
            self.num_out_conv = 28
        elif self.num_grid == 512:
            self.num_out_conv = 60

        self.num_outputs = num_outputs
        self.fc_size = 512

        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.num_feature, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * self.num_out_conv * self.num_out_conv, self.fc_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_size, num_outputs),
            torch.nn.LogSoftmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, 64 * self.num_out_conv * self.num_out_conv)  # 展开 拉成一维 输入到全连接层
        out = self.fc(out)
        return out

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.Tensor(batch.action).float().to(device)
        rewards = torch.Tensor(batch.reward).to(device)
        humans = torch.Tensor(batch.human).to(device)

        pred = online_net(states)
        next_pred = target_net(next_states)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + humans*gamma * next_pred.max(1)[0]
    
        loss=F.smooth_l1_loss(pred, target.detach())
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss,torch.sum(pred),torch.sum(target.detach())

    def get_action(self, input):
        input=input.view(-1,self.num_feature,self.num_grid,self.num_grid)

        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)

        del input, qvalue, _

        return action.cpu().detach().numpy()[0]
