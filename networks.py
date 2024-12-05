import torch.nn as nn

# Размер латентного вектора для генератора
nz = 100

# Размер карт признаков для генератора
ngf = 64

# Размер карт признаков для дискриминатора
ndf = 64

# Размер изображений
image_size = 28  # Для MNIST

# Количество каналов на входе (1 для изображений MNIST)
nc = 1

class Generator(nn.Module):
  '''
  Класс генератора. Принимает тензор размера 100 на вход, на выходе тензор размера 784. Полносвязная прямая сеть.
  Цель генератора в том, чтобы сгенерировать выходной тензор, который максимально похож на тензор из MNIST 
  '''

  def __init__(self):
    super().__init__()   
                                                            # Инициализация базового класса, чтобы унаследованный класс получил все свойства и методы от базового класса 
    self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # 1x1x100 -> 4x4x512
                                nn.BatchNorm2d(ngf * 8),
                                nn.ReLU(True))                               
      
    self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 4x4x512 -> 8x8x256
                                nn.BatchNorm2d(ngf * 4),
                                nn.ReLU(True))
    
    self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 8x8x256 -> 16x16x128
                                nn.BatchNorm2d(ngf * 2),
                                nn.ReLU(True))
    
    self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 16x16x128 -> 32x32x64
                                nn.BatchNorm2d(ngf),
                                nn.ReLU(True))
    
    self.output = nn.Sequential(nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),  # 32x32x64 -> 28x28x1
                                nn.Tanh())                                        # Преобразуем к диапазону (-1, 1))    

  def forward(self, x):   
    #print(f"Generator input shape: {x.shape}")                                                     # Прогонка тензоров через сеть. Важный момент: PyTorch автоматически обрабатывает тензоры параллельно
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.output(x)
    #print(f"Generator output shape: {x.shape}")
    return x
  
class Discriminator(nn.Module):
  '''
  Класс дискриминатора. Принимает тензор размера 784 на вход, на выходе тензор размера 1, обозначающий вероятность принадлежности классу 1 (реального изображения). Полносвязная прямая сеть. 
  Цель дискриминатора в том, чтобы отличать реальные изображения от сгенерированных.
  '''

  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 28x28x1 -> 14x14x64
        nn.LeakyReLU(0.2, inplace=True))
    
    self.layer2 = nn.Sequential(
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 14x14x64 -> 7x7x128
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True))
    
    self.layer3 = nn.Sequential(
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 7x7x128 -> 4x4x256
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True))
    
    # self.layer4 = nn.Sequential(
    #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 4x4x256 -> 2x2x512
    #     nn.BatchNorm2d(ndf * 8),
    #     nn.LeakyReLU(0.2, inplace=True))
    
    self.output = nn.Sequential(
        nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),  # 4x4x256 -> 1x1x1
        nn.Sigmoid())

  def forward(self, x):
    #print(f"Discriminator input shape: {x.shape}")
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    #x = self.layer4(x)
    x = self.output(x)
    #print(f"Discriminator output: {x.shape}")
    return x
  