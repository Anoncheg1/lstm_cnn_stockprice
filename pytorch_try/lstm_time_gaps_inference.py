# from __future__ import print_function
# one step - Striding windows in strided timeline
import torch
import torch.nn as nn
import numpy as np
import matplotlib

from lstm_time_gaps import Sequence
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# TODO: use data as input before future.
# calc loss without test_inpit only future

from lstm_time_gaps import Sequence, dtype


def main():
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # LOAD PARAMETERS
    data: np.array = torch.load('traindata_ya_batch.pt')
    data_offset = data[2]
    # batch_size = data[3]

    # LOAD DATA and make training set
    data_orig: np.array = torch.load('traindata_ya_orig.pt')  # steps, time/price
    print("batches:", len(data_orig), len(data_orig[0]), type(data_orig))

    # LOAD MODEL
    model = Sequence(input_size=2, hidden_size=200, levels=2).to(dtype)
    model.load_state_dict(torch.load('a2.pt'))
    model.eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # criterion = nn.MSELoss()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     criterion = criterion.cuda()

    model.new_hidden(1, "cpu")
    # print(list(model.parameters()))

    print('begin to predict, no need to track gradient here')
    res_t = []
    res_p = []
    future_time = np.arange(data_orig.shape[0]) / data_orig.shape[0] + 1
    time_line = np.concatenate([data_orig[:, 0], future_time])
    c = data_orig.shape[0] // data_offset - 1
    with torch.no_grad():
        # seq.new_hidden(1)
        for st in range(data_orig.shape[0] - 1 + data_offset*c, 0, -data_offset)[::-1]:
            print(st)
            o_place = st + data_offset
            # price_date = torch.from_numpy(data[0][0][0]).to(torch.float16).unsqueeze(0)  # .double() # [1,2]
            if st < 500:
                price_date = torch.from_numpy(data_orig[st]).to(dtype).unsqueeze(0)  # .double() # [1,2]
            else:
                price_date = torch.Tensor([time_line[st], out]).to(dtype).unsqueeze(0)
            # print(price_date)
            # next_price = torch.from_numpy(data_orig[o_place])[1].to(dtype).view(1, 1, 1)  # .double()
            # if torch.cuda.is_available():
            #     price_date = price_date.cuda()
            #     next_price = next_price.cuda()
            out = model(price_date)
            res_t.append(time_line[o_place])  # time
            res_p.append(out.item())  # result price

    plt.plot(data_orig[:, 0], data_orig[:, 1], 'r', linewidth=2.0)  # without first

    # now_time = time[:res_test_prices]
    plt.plot(res_t, res_p, 'g', linewidth=2.0)
    plt.show()
    exit(0)

    # for i in range(test_input.size(0)):
    #         out: torch.Tensor = seq(test_input[i])  # forward - reset state
    #         loss = criterion(out.squeeze(), test_target[i])
    #         if i % 10 == 0:
    #             print(i, 'test loss:', loss.item())
    #
    # # PREDICT FUTURE
    # # load original
    # data: np.array = torch.load('traindata_ya_orig.pt')  # steps, time/price
    # print("batches:", len(data), len(data[0]), type(data), data.shape)
    # # train = :sl
    # # test = sl:
    # offset = 100  # fixed
    # steps_before = 8
    # steps_after = 5
    # where = list(range(830, 1200, 100))  # where predict
    # test_pred = []
    # for w in where:
    #     b = w - offset * steps_before + 1
    #     a = w + offset * steps_after
    #     t_p = data[b:a:offset, :]
    #     t_p = torch.from_numpy(t_p).unsqueeze(1).double()
    #     if torch.cuda.is_available():
    #         t_p = t_p.cuda()  # GPU
    #     test_pred.append(t_p)  # 110
    #
    # # print(test_pred.size())
    # yy = []
    # with torch.no_grad():
    #     predictions = []
    #     for t_p in test_pred:  # = len(where)
    #         # print("t_p", t_p[:steps_before, :, :].size())  # step, batch, price/time
    #         seq.new_hidden(1)
    #         for b_i in t_p[:steps_before]:  # batch, price/time
    #             pred: torch.Tensor = seq(b_i)
    #             # print(b_i.shape)
    #             # print(pred)
    #
    #         # FUTURE
    #         for b_i in t_p[-steps_after:, :, 0]:
    #             b_i = b_i.squeeze()
    #             # print("future", b_i.size())
    #             pred = pred.squeeze()
    #             # print("pred", pred.size())
    #             inp = torch.stack([b_i, pred]).unsqueeze(0)
    #             # print(inp.size())
    #             pred = seq(inp)  # 1, time/price
    #             # print(pred)
    #             predictions.append(pred)
    #     yy.append(predictions)
    #
    #     # loss = criterion(pred, test_pred[:, 1].view(-1, 1))
    #     # print('test loss:', loss.item())
    #     # GPU
    #     # if torch.cuda.is_available():
    #     #     pred = pred.cpu()
    #     # y = pred.detach().numpy()
    #     # yy.append(y)
    #
    #     # y = np.concatenate(y)
    #     # input2 = striding_windows_reverse(input2)
    #     # data2 = striding_windows_reverse(list(data))
    #
    #     # # DRAW THE RESULT
    #
    # plt.figure(figsize=(30, 10))
    # plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    # plt.xlabel('x', fontsize=20)
    # plt.ylabel('y', fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # # print(yy)
    # # print(input2)
    #
    # plt.plot(data[:, 0], data[:, 1], 'r', linewidth=2.0)
    # # plt.plot(np.arange(len(data2)), np.array(data2), 'g', linewidth=2.0)
    # for i, y in enumerate(yy):  # = len(where)
    #     a = where[i] + offset * steps_after
    #     future_time = np.arange(2000) / 2000 + 1
    #     time = np.concatenate([data[:, 0], future_time])
    #     now_time = time[where[i]:a:offset]
    #     print(where[i], a)
    #     # print(now_time)
    #     plt.plot(now_time, y, 'g', linewidth=2.0)
    # # plt.plot(np.arange(train_l + len(y) - future, train_l + len(y)), y[-future:], 'b' + ':', linewidth=2.0)
    # plt.savefig('predict%d.pdf' % 1)
    # plt.show()
    # # draw(y[1], 'g')
    # # # draw(y[2], 'b')
    # # # draw(y[3], 'b')
    # plt.savefig('predict%d.pdf' % 1)
    # plt.close()


if __name__ == '__main__':
    main()
