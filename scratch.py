import torch as th
import matplotlib.pyplot as plt


def steep_sigmoid(x, steepness=50, bias=0):
    # approximates a hard threshold set at "bias"
    # above "bias" are mapped to 1 and values below mapped to 0
    return th.sigmoid((x - bias) * steepness)

alpha = th.rand(3, 3)

# round to 2 decimals to make life simpler
alpha = th.round(alpha * 100) / 100

step = 0.01
data = th.arange(0, 1 + step, step=step)
# data = th.arange(-2, 2, step=0.05)
data_out = steep_sigmoid(data)

msg_send_binary = steep_sigmoid(alpha)

alpha_out = alpha * msg_send_binary

print(alpha)
print(msg_send_binary)
print(alpha_out)
# plt.plot(data, data)
plt.plot(data, data_out)
plt.savefig("scratch.png")


# # sigmoid = th.nn.Sigmoid()
# alpha_filtered = steep_sigmoid(alpha)


# print('\n breakpoint ')
# __import__('ipdb').set_trace(context=3)
