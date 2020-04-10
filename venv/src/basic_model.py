import torch

class BasicModule(torch.nn.Module):
    """
    encapsulate the nn.Module to providing both load and save functions
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
    # 加载模型
    def load(self, path, device):
        print('starting to LOAD the ${}$ Model from {} within the {} device'.format(self.model_name, path, device))
        # 如果当前的设备是cpu，则加载到cpu上
        if device == torch.device('cpu'):
            self.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.load_state_dict(torch.load(path))
    # 保存模型，name为文件路径的名称
    def save(self, name=None):
        assert name is not None, 'please specify the path name to save the module'
        # 打开文件将模型参数写入文件中保存，文件如果已经存在则进行覆盖
        with open(name, 'wb') as file:
            # 仅仅保存模型参数.pt形式
            torch.save(self.state_dict(), file)
        print('starting to SAVE the ${}$ Model to ${}$\n'.format(self.model_name, name))
