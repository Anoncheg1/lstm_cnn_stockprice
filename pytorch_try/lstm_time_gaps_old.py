# from __future__ import print_function
# one step - Striding windows in strided timeline
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
from data_augum import striding_windows_reverse

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# TODO: use data as input before future.
# calc loss without test_inpit only future


class Sequence(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, levels: int):
        super(Sequence, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.levels = levels
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.levels, dropout=0.2)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input: torch.Tensor, future: torch.Tensor = None):  # 4, 99
        outputs = []
        self.hidden = (
            torch.rand(self.levels, input.size(1), self.hidden_size, dtype=torch.double),  # layers, batch, hidden
            torch.rand(self.levels, input.size(1), self.hidden_size, dtype=torch.double))
        if torch.cuda.is_available():
            self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())

        output = torch.rand(1, input.size(1), 1, dtype=torch.double).cuda()
        # parallel in mini-batch
        for i, input_t in enumerate(input):  # 40 of [99]
            # print(input_t.size())
            input_t: torch.Tensor = input_t.unsqueeze(0)  # [1, 99, 2]

            h_t, self.hidden = self.lstm(input_t, self.hidden)
            output = self.linear(h_t)  # [1, 99, 1]
            if future is None:
                outputs += [output]  # 40 list of [1, 99, 1]
            # else:
            #     print("w2", output)
        if future is not None:
            for f_i in future:  # future timestamps
                f_i = f_i.unsqueeze(0).unsqueeze(0)
                # print(f_i.size())
                # print(output.size())
                output = torch.stack([f_i, output], dim=2).squeeze(-1)
                # print(output)
                # print(output.size())
                h_t, self.hidden = self.lstm(output, self.hidden)
                output = self.linear(h_t)
                outputs += [output]
        outputs = torch.stack(outputs).squeeze()  # [40, 1, 99, 1] -> [40, 99]
        return outputs


def main():
    STEPS = 1
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    data: np.array = torch.load('traindata_ya_batch.pt')
    print("batches:", len(data), len(data[0]), type(data), data.shape)
    # [100, 1000] we use 97 inputes for train and 3 for test [97, 999]
    # 100 batches - we learn one function at all batches
    sl = data.shape[1] - data.shape[1] // 10  # test amount
    print(sl)

    # # (2, 300, 10, 2) # train/test, steps, batchs, time/price
    input = torch.from_numpy(data[0, :sl, :]).double()  # range (-1 1) first sl, without last 1
    print("input", input.size())
    # [100, 1000] we use 97 inputes for train and 3 for test [97, 999]
    target = torch.from_numpy(data[1, :sl, :, 1]).squeeze().double()  # without first 1
    print("train", target.size())

    # future = 300
    test_input = torch.from_numpy(data[0, sl:, :]).double()
    print("test", len(test_input), len(test_input[0]))
    test_target = torch.from_numpy(data[1, sl:, :, 1]).squeeze().double()  # second sl, without first

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        input = input.cuda()  # GPU
        target = target.cuda()  # GPU
        test_input = test_input.cuda()
        test_target = test_target.cuda()

    # build the model
    seq = Sequence(input_size=2, hidden_size=200, levels=3)
    seq.double()
    seq = seq.to(device)  # GPU
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.3)
    # begin to train
    input2 = input.reshape(5, 108, 5, 2)
    target2 = target.reshape(5, 108, 5)
    # print(target.shape)
    # for i1 in range(STEPS):
    for i, _ in enumerate(input2):
        # print('STEP: ', i1, i)

        def closure():
            optimizer.zero_grad()
            out = seq(input2[i])  # forward - reset state
            # out = seq(input)  # forward - reset state
            # print("out", out)
            loss = criterion(out, target2[i])
            # loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        # cannot predict here - internal error
    print('begin to predict, no need to track gradient here')

    with torch.no_grad():
        # print(test_input)
        out = seq(test_input)  # steps, batchs, time/price
        if len(out.shape) == 1:
            out = out.view(-1, 1)
        # print("out", out)
        loss = criterion(out, test_target)
        print('test loss:', loss.item())
        # GPU
        # if torch.cuda.is_available():
        #     pred = pred.cpu()
        #     input2 = input.cpu()
        # y = pred.detach().numpy()
        # input2 = input2.detach().numpy()

    # PREDICT FUTURE
    # load original
    data: np.array = torch.load('traindata_ya_orig.pt')  # steps, time/price
    sl = data.shape[0] - data.shape[0] // 5  # test amount
    print("batches:", len(data), len(data[0]), type(data), data.shape)
    # train = :sl
    # test = sl:
    offset = 100  # fixed
    steps_before = 10
    steps_after = 5
    where = list(range(1030, 2030, 100))  # where predict
    test_pred = []
    for w in where:
        b = w - offset * steps_before
        a = w + offset * (steps_after + 1)
        t_p = data[b:a:offset, :]
        t_p = torch.from_numpy(t_p).unsqueeze(1).double()
        if torch.cuda.is_available():
            t_p = t_p.cuda()  # GPU
        test_pred.append(t_p)  # 110

    # print(test_pred.size())
    yy = []
    with torch.no_grad():
        for t_p in test_pred:  #  = len(where)
            # print("t_p", t_p[:steps_before, :, :])
            pred = seq(t_p[:steps_before, :, :], future=t_p[-steps_after:, :, 0])  # steps, 1, time/price
            if len(pred.shape) == 1:
                pred = pred.view(-1, 1)
            # loss = criterion(pred, test_pred[:, 1].view(-1, 1))
            # print('test loss:', loss.item())
            # GPU
            if torch.cuda.is_available():
                pred = pred.cpu()
            y = pred.detach().numpy()
            yy.append(y)

        # y = np.concatenate(y)
        # input2 = striding_windows_reverse(input2)
        # data2 = striding_windows_reverse(list(data))

        # # DRAW THE RESULT

    plt.figure(figsize=(30, 10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # print(yy)
    # print(input2)

    plt.plot(data[:, 0], data[:, 1], 'r', linewidth=2.0)
    # plt.plot(np.arange(len(data2)), np.array(data2), 'g', linewidth=2.0)
    for i, y in enumerate(yy):  #  = len(where)
        a = where[i]+offset * steps_after
        future_time = np.arange(2000) / 2000 + 1
        time = np.concatenate([data[:, 0], future_time])
        now_time = time[where[i]:a:offset]
        # print(now_time)
        plt.plot(now_time, y, 'g', linewidth=2.0)
    # plt.plot(np.arange(train_l + len(y) - future, train_l + len(y)), y[-future:], 'b' + ':', linewidth=2.0)
    plt.savefig('predict%d.pdf' % 1)
    plt.show()
    # draw(y[1], 'g')
    # # draw(y[2], 'b')
    # # draw(y[3], 'b')
    plt.savefig('predict%d.pdf' % 1)
    plt.close()


if __name__ == '__main__':
   main()
