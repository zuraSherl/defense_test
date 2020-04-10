import torch
import torch.nn as nn

# k-WTA激活函数的基础模块
class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        # sparse_ratio：稀疏比，默认为0.5
        # preact：激活前的值
        # act：激活后的值
        self.sr = sparse_ratio
        self.preact = None
        self.act = None
    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()
        return hook
    # 记录preact和act的值
    def record_activation(self):
        self.register_forward_hook(self.get_activation())

# 根据稀疏比进行k-WTA激活（用于线性层）
class Sparsify1D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify1D, self).__init__()
        # sparse_ratio：稀疏比，默认为0.5
        self.sr = sparse_ratio
    # 通过稀疏比进行像素保留
    def forward(self, x):
        # x：输入图像，shape为N*size
        # k：需要保留的像素数=一张图像的总像素数*稀疏比
        k = int(self.sr*x.shape[1])
        # topval：获取x中最小的需要保留的像素值
        topval = x.topk(k, dim=1)[0][:, -1]
        # topval：将最小值扩展成与x相同结构的tensor
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1,0)
        # 如果x中的像素值大于最小需要保留的像素值，则返回1，否则返回0
        comp = (x>=topval).to(x)
        # 返回保留的像素值，其他的像素置为0
        return comp*x

# 根据稀疏比进行k-WTA激活（每一层）
class Sparsify2D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D, self).__init__()
        # sparse_ratio：稀疏比，默认为0.5
        self.sr = sparse_ratio
        self.preact = None
        self.act = None
    def forward(self, x):
        # 输入x的shape为N*C*H*W
        # layer_size：每一层的H*W总的像素数
        layer_size = x.shape[2]*x.shape[3]
        # k：需要保留的像素数=每一层总像素数*稀疏比
        k = int(self.sr*layer_size)
        # 将x的shape转换为N*C*(H*W)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        # topval：获取H*W中最小的需要保留的像素值
        topval = tmpx.topk(k, dim=2)[0][:,:,-1]
        # 将最小值扩展成与x相同结构的tensor
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2,3,0,1)
        # 如果x中的像素值大于最小需要保留的像素值，则返回1，否则返回0
        comp = (x>=topval).to(x)
        # 返回保留的像素值，其他的像素置为0
        return comp*x

# 根据稀疏比进行k-WTA激活（每个通道）
class Sparsify2D_vol(SparsifyBase):
    '''cross channel sparsify'''
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_vol, self).__init__()
        # sparse_ratio：稀疏比，默认为0.5
        self.sr = sparse_ratio
    def forward(self, x):
        # 输入x的shape为N*C*H*W
        # 每个通道总的像素数
        size = x.shape[1]*x.shape[2]*x.shape[3]
        # k：需要保留的像素数=每一通道的总像素数*稀疏比
        k = int(self.sr*size)
        # tmpx：将x的shape转换为N*(C*H*W)
        tmpx = x.view(x.shape[0], -1)
        # topval：获取C*H*W中最小的需要保留的像素值
        topval = tmpx.topk(k, dim=1)[0][:,-1]
        # 将最小值扩展成与x相同结构的tensor
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        # 如果x中的像素值大于最小需要保留的像素值，则返回1，否则返回0
        comp = (x>=topval).to(x)
        # 返回保留的像素值，其他的像素置为0
        return comp*x

# 定义swish激活函数
class Swish(nn.Module):
    def __init__(self, b = 1):
        super(Swish, self).__init__()
        self.b = b
    def forward(self, x):
        x = x * torch.sigmoid(self.b * x)
        return x

# 定义有界ReLU函数
class BReLU(nn.Module):
    def __init__(self, t = 6):
        super(BReLU, self).__init__()
        self.t = t
    def forward(self, x):
        x = torch.clamp(x, min = 0, max = self.t)
        return x

sparse_func_dict = {
    'reg':Sparsify2D,  #top-k value，按照每一层进行稀疏比k-WTA激活
    'vol':Sparsify2D_vol,  #cross channel top-k，按照每一个通道进行稀疏比k-WTA激活
    'relu':nn.ReLU, #ReLU激活函数
    'leaky_relu':nn.LeakyReLU, #LeakyReLU激活函数，大于等于0取原值，小于0取ax
    'p_relu':nn.PReLU, #PReLU激活函数，a的值可调整
    'relu6':nn.ReLU6, #ReLU6激活函数，output = min(max(0,x), 6)
    'rrelu':nn.RReLU, #RReLU激活函数，大于等于0取原值，小于0取mx，m~U(l,u),u~[0,1)
    'elu':nn.ELU, # ELU激活函数，大于等于0取原值，小于0取a*(exp(x)-1)，a默认为1
    'selu':nn.SELU, # SELU激活函数，大于0取mx，小于0取ma(exp(x)-1),m默认为1.0507，a默认为1.67326
    'softplus':nn.Softplus, # Softplus激活函数，output=log(1+exp(x))
    'threshold':nn.Threshold, # Threshold激活函数，y=x ,if x>=threshold y=value,if x<threshold
    'swish':Swish, # Swish激活函数，f(x)=x*sigmod(b*x)
    'brelu':BReLU # 对relu6的扩展，output = min(max(0,x), t)，参数t可选
}
