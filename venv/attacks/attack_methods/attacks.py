from abc import ABCMeta
from abc import abstractmethod

# ABCMeta：用来生成抽象基础类的元类。由它生成的类可以被直接继承
# abstractmethod：表明抽象方法的生成器
class Attack(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def perturbation(self):
        print("Abstract Method of Attacks is not implemented")
        raise NotImplementedError