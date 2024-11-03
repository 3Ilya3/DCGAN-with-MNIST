import torch.nn as nn

class Generator(nn.Module):
  '''
  Класс генератора. Принимает тензор размера 100 на вход, на выходе тензор размера 784. Полносвязная прямая сеть.
  Цель генератора в том, чтобы сгенерировать выходной тензор, который максимально похож на тензор из MNIST 
  '''
  
  def __init__(self):
    super().__init__()                                                           # Инициализация базового класса, чтобы унаследованный класс получил все свойства и методы от базового класса 
    self.layer1 = nn.Sequential(nn.Linear(in_features=100, out_features=256),
                                nn.LeakyReLU())                                  # Почему LeakyReLU()? Функция помогает избежать проблемы "затухающих градиентов", когда градиенты равны 0
    self.layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=784),
                                nn.Tanh())                                       # Tanh нужен для преобразования значений к пикселям (-1;1), чтобы соответствовать нормализации

  def forward(self, x):                                                          # Прогонка тензоров через сеть. Важный момент: PyTorch автоматически обрабатывает тензоры параллельно
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x
  
class Discriminator(nn.Module):
  '''
  Класс дискриминатора. Принимает тензор размера 784 на вход, на выходе тензор размера 1, обозначающий вероятность принадлежности классу 1 (реального изображения). Полносвязная прямая сеть. 
  Цель дискриминатора в том, чтобы отличать реальные изображения от сгенерированных.
  '''

  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Linear(in_features=784, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                nn.Sigmoid())                                  # Sigmoid нужен для преобразования значений в вероятность (0;1). Хорошо сочетается с бинарной кросс-энтропийной функцией потерь
    
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x
  