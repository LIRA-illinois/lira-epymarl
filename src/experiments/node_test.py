
import torch as th

a1 = th.tensor([1, 2, 3])

a1.cuda()
print("Succsesfully put tensor on device")
