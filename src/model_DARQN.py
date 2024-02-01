import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention.SEAttention import SEAttention
from .attention.ECAAttention import ECAAttention
from .config_DARQN import gamma, device, batch_size, sequence_length, burn_in_length, action_space


# torch.manual_seed(0)

class DARQN(nn.Module):
    def __init__(self, num_grid, num_feature, num_outputs):
        super(DARQN, self).__init__()
        self.num_grid = num_grid
        self.num_feature = num_feature
        self.num_inputs = self.num_feature * self.num_grid * self.num_grid
        if self.num_grid == 48:
            self.num_out_conv = 2
        elif self.num_grid == 64:
            self.num_out_conv = 4
        elif self.num_grid == 128:
            self.num_out_conv = 12
        elif self.num_grid == 192:
            self.num_out_conv = 20
        elif self.num_grid == 216:
            self.num_out_conv = 23
        elif self.num_grid == 224:
            self.num_out_conv = 24
        elif self.num_grid == 240:
            self.num_out_conv = 26
        elif self.num_grid == 256:
            self.num_out_conv = 28
        elif self.num_grid == 512:
            self.num_out_conv = 60
        self.num_outputs = num_outputs
        self.max_length=128
        self.hidden_size=256
        self.at_size = 328
        self.fc_size=256

    
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_feature, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=self.at_size, kernel_size=3, stride=1),
            nn.ReLU()
        )


        self.se_att=SEAttention(channel=self.at_size,reduction=8)

     
        self.fc = torch.nn.Sequential(
            nn.Linear(self.at_size * self.num_out_conv * self.num_out_conv, self.fc_size),
            nn.ReLU()
        )

   
        self.rnn = nn.LSTM(input_size=self.fc_size, hidden_size=self.hidden_size, batch_first=True)

     
        self.out = torch.nn.Sequential(

            nn.Linear(self.hidden_size,self.num_outputs),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

   
    def forward(self, x, hidden=None, batch_size=1, sequence_length=1):
      
        x = x.view(-1, self.num_feature, self.num_grid, self.num_grid)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out=self.se_att(out)

   
        out = out.view(out.size(0), -1)
        out=self.fc(out)

        out = out.reshape(batch_size, sequence_length, self.fc_size)

        if hidden is not None:
            out, hidden = self.rnn(out, hidden)
            # print('if', out.shape, hidden[0].shape, x.shape)
        else:
            out, hidden = self.rnn(out)
            # print('else', out.shape, hidden[0].shape, x.shape)

        out=self.out(out)

        return out, hidden

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        def slice_burn_in(item):
            return item[:, burn_in_length:, :]

        states = torch.stack(batch.state).view(batch_size, sequence_length, online_net.num_inputs).to(device)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, online_net.num_inputs).to(device)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long().to(device)
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1).to(device)
        humans = torch.stack(batch.human).view(batch_size, sequence_length, -1).to(device)
      
        states = states.unsqueeze(0)
        next_states = next_states.unsqueeze(0)
       
        pred, _ = online_net(states, batch_size=batch_size, sequence_length=sequence_length)
        next_pred, _ = target_net(next_states, batch_size=batch_size, sequence_length=sequence_length)

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        a_0 = actions.shape[0]
        a_1 = actions.shape[1]
        actions = torch.max(actions, 2)[1]
        actions = actions.view(a_0, a_1, 1)

        rewards = slice_burn_in(rewards)
        humans = slice_burn_in(humans)

        pred = pred.gather(2, actions)

  
        target = rewards + humans * gamma * next_pred.max(2, keepdim=True)[0]

        loss = F.smooth_l1_loss(pred, target.detach())
        # loss = torch.log(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del states,next_states,rewards,humans,a_0,a_1,actions,_,next_pred

        return loss,torch.sum(pred),torch.sum(target.detach())

    def get_action(self, state, hidden):

        state = state.unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden)

        _, action = torch.max(qvalue, 2)

        return action.detach().cpu().numpy()[0][0], hidden
