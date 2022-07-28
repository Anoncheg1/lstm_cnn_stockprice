import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
# own
from data_augum import augum


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int, levels=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=levels)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # initial and working hidden
        self.hidden = (torch.rand(levels, batch_size, HIDDEN_DIM),
                       torch.rand(levels, batch_size, HIDDEN_DIM))

    def forward(self, inp):


        # print(input)

        out, _ = self.lstm(inp, self.hidden)
        return self.linear(out)


        # for i in range(future):
        #     if y is not None and random.random() > 0.5:
        #         output = y[:, [i]]  # teacher forcing
        #     h_t, c_t = self.lstm(output, (h_t, c_t))
        #     output = self.linear(h_t)
        #     outputs += [output]
        # outputs = torch.stack(outputs, 1).squeeze(2)
        # return outputs


INPUT_SIZE = 1
HIDDEN_DIM = 50
OUTPUT_SIZE = 1
BATCH_SIZE = 1
p = '/home/py/GAZP_200103_200105.csv'
# data = augum(p, [7])
data = np.array(list(range(30)))
data_min = np.nanmin(data, axis=0)
data_max = np.nanmax(data, axis=0)
data = (np.array(data) - data_min) / (data_max - data_min)
print(data)

# SEED
torch.manual_seed(1)
# MODEL
model: nn.Module = Model(INPUT_SIZE, HIDDEN_DIM, OUTPUT_SIZE, BATCH_SIZE, levels=1)
# GRADIENT OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=0.001)
# LOSS MSE FUNCTION
loss_func = nn.MSELoss()

# EPOCHS
for step in range(60):
    # minibatch
    for i, x in enumerate(data[:len(data)-1]):  # without last ----
        inputs_t = torch.tensor(x, dtype=torch.float32).view(1, BATCH_SIZE, -1)
        # without first ----
        y = torch.tensor(data[1:][i], dtype=torch.float32).view(1, BATCH_SIZE, -1)

        model.zero_grad()
        optimizer.zero_grad()
        # PREDICT
        predict = model(inputs_t)  # without last ----
        # CALC LOSS
        loss = loss_func(predict, y)
        #
        loss.backward(retain_graph=True)
        optimizer.step()

with torch.no_grad():
    for i, x in enumerate(data):
        if i > 10:
            break
        t = torch.tensor(x, dtype=torch.float32).view(1, BATCH_SIZE, -1)
        print(model(t), t)

# print(a)
#
# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# hidden = (torch.randn(1, 1, 1),
#           torch.randn(1, 1, 1))
#
# for epoch in range(10):
#     for i, x in enumerate(data):
#         # 1 clear grads
#         model.zero_grad()
#
#         # 2 convert to tensor
#         # xt = torch.tensor(x, dtype=torch.long)
#
#         # 3 forward pass
#         y = model(x)
#
#         # 4 loss
#         loss = loss_function(y, data(x-1))
#         # calc gradients
#         loss.backward()
#         # apply grads to weights
#         optimizer.step()
#
#
# with torch.no_grad():
#     inputs = data[0]
#     res = model(inputs)
#     print(res)


# print(f"Trainable parameters: {params:,}")
# print(out)
# print(hn)
# print(cn)
