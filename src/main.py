import torch
from cuda_extension._ops import reduce_sum

if __name__ == "__main__":
    size = int(1e3)
    input_ = torch.randn(size).cuda()
    print(reduce_sum(input_))