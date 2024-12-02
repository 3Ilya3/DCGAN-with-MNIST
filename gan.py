import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter

import networks as n

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAN():

  def __init__(self, learning_rate, model_path=None):
    self.learning_rate = learning_rate
    self.generator = n.Generator().to(device)
    self.discriminator = n.Discriminator().to(device)
    '''
    После каждой эпохи мы генерируем 100 изображений (случайных шумовых векторов длиной 100 из стандартного нормального распределения. Тензор, состоящий из 100 тензоров).
    Сохраняем выходные изображения в списке test_progression, чтобы отслеживать и визуализировать прогресс генератора на протяжении обучения. 
    '''
    self.test_noises = torch.randn(100,1,100, device=device)
    self.test_progression = []    

    # Оптимизаторы
    self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
    self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate) 

    self.current_epoch = 0

    if model_path is not None:
            self.load_model(model_path)

    #self.writer = SummaryWriter(log_dir='logs/gan_experiment')                             

  def forward(self, z):
    """
    Передача тензора (вектор шума) генератору.
    Генерация изображения.
    """
    return self.generator(z)

  def generator_step(self, x):
    
    # Создание случайного шума для генерации изображения. x.shape[0] это количество примеров в текущем батче. Как и везде находится в 1 размерности
    z = torch.randn(x.shape[0], 1, 100, device=device)

    # Генерация изображения
    generated_imgs = self.forward(z)

    # Классифицируем сгенерированные изображения через бинарный классификатор, он же наш дискриминатор. 
    # Результат - тензор размера батча с вероятностями каждого изображения, что оно реальное.
    # squeeze() удаляет все размерности равные 1, для удобства, чтобы у нас просто был тензор с вероятностями 
    d_output = torch.squeeze(self.discriminator(generated_imgs))

    '''
    Важный момент: 
    Мы хотим максимизировать ошибку дискриминатора для поддельных изображений. 
    Поскольку PyTorch предназначен для минимизации функций потерь (а не их максимизации), чтобы добиться этого, мы перевернули метки (0 -> 1).
    Генератор минимизирует BCELoss с метками 1, чтобы добиться той же цели.
    '''
        
    # Бинарная кросс энтропия, она же функция потери. У нас есть два распределения: d_ouput - предсказанная вероятность и метки равные 1 (метка реального изображения, ones создает тензор единиц).
    # Считаем меру разницы распределений, чем меньше, тем они похожи.
    g_loss = nn.BCELoss()(d_output, torch.ones(x.shape[0], device=device))

    # Обратное распространение и обновление весов
    self.g_optimizer.zero_grad()
    g_loss.backward()
    self.g_optimizer.step()

    return g_loss.item()

  def discriminator_step(self, x):
    
    # Прогоняем реальные изображения из батча. loss_real будет низкой, если дискриминатор правильно определяет их как настоящие
    d_output_real = torch.squeeze(self.discriminator(x))
    loss_real = nn.BCELoss()(d_output_real, torch.ones(x.shape[0], device=device))

    # Прогоняем поддельные изображения, сгенерированные генератором из шума. loss_fake будет низкой, если дискриминатор правильно определяет их как поддельные
    z = torch.randn(x.shape[0], 1, 100, device=device)
    generated_imgs = self.forward(z)
    d_output_fake = torch.squeeze(self.discriminator(generated_imgs))
    loss_fake = nn.BCELoss()(d_output_fake, torch.zeros(x.shape[0], device=device))  # zeros - тензор с нулями

    # Дискриминатор учится сразу на двух задачах, на выявление реального и поддельного изображения. Возвращает общую потерю для дискриминатора. Будем минимизировать ее
    d_loss = loss_real + loss_fake
     
    # Обратное распространение и обновление весов
    self.d_optimizer.zero_grad()
    d_loss.backward()
    self.d_optimizer.step()

    return d_loss.item()
  
  '''
  Минимаксная игра это как раз таки: 
  Генератор стремится минимизировать свою потерю, которая в свою очередь максимизирует потерю дискриминатора.
  Дискриминатор стремится минимизировать свою потерю.
  Задача в том, чтобы прийти к равновесию: 
  Так как вероятность для реальных и сгенерированных данных становится равной, то есть их распределения не отличаются, 
  дискриминатор начинает давать каждому примеру (и реальным и сгенирированным) вероятность 0.5, что он принадлежит к реальному распределению.
  Хорошо написано у Гудфеллоу: https://arxiv.org/pdf/1406.2661
  '''

  def train(self, dataloader, num_epochs):
        
        current_epoch = self.current_epoch
        total_epoch = num_epochs + current_epoch

        for epoch in range(current_epoch, total_epoch + 1):
            
            with tqdm(dataloader, desc=f"Epoch {epoch}/{total_epoch}", unit="batch") as pbar:
                # _ это метки цифр (0-9), в случае GAN метки для реальных данных не нужны. Мы занимаемся генерацией, а не классификацией.
                for batch, _ in pbar: 
                    batch = batch.to(device)

                    # Обучение генератора и дискриминатора
                    g_loss = self.generator_step(batch)
                    d_loss = self.discriminator_step(batch)

                    # Логирование потерь в TensorBoard
                    #self.writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
                    #self.writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)

                    pbar.set_postfix(G_loss=g_loss, D_loss=d_loss)

            # Сохранение прогресса
            epoch_test_images = self.forward(self.test_noises)
            self.test_progression.append(epoch_test_images.detach().cpu().numpy())


            if epoch % 100 == 0:
                self.current_epoch = epoch
                #self.save_model(f'models/model_{epoch}')
                self.visualize_images(epoch)
                

            if epoch % 500 == 0:
               self.current_epoch = epoch
               self.save_model(f'models/model_{epoch}')
               #self.visualize_images(epoch)

        self.current_epoch = total_epoch

  def visualize_images(self, epoch):
      #plt.ion()
      nrow, ncol = 3, 8  # Размеры сетки
      fig = plt.figure(figsize=((ncol + 1) * 2, (nrow + 1) * 2))
      fig.suptitle(f'Epoch {epoch}', fontsize=30)
      gs = gridspec.GridSpec(nrow, ncol,
                             wspace=0.0, hspace=0.0,
                             top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                             left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
      # Выводим изображения
      for i in range(nrow):
          for j in range(ncol):
              img_idx = i * ncol + j
              img = np.reshape(self.test_progression[-1][img_idx], (28, 28))
              ax = plt.subplot(gs[i, j])
              ax.imshow(img, cmap='gray')
              ax.axis('off')
      
      plt.savefig(f'images/image_of_epoch_{epoch}.png')
      #plt.show()
      #plt.pause(0.001)  # Обновляем визуализацию
      #plt.pause(3)
      plt.close(fig)  # Закрываем текущее окно, чтобы оно не оставалось открытым

  def save_model(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.g_optimizer.state_dict(),
            'optimizer_d_state_dict': self.d_optimizer.state_dict(),
            'test_progression': self.test_progression,
            'current_epoch': self.current_epoch
        }, path)
        print(f'Model saved as {path} at epoch {self.current_epoch}')

  def load_model(self, path):
      checkpoint = torch.load(path)
      self.generator.load_state_dict(checkpoint['generator_state_dict'])
      self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
      self.g_optimizer.load_state_dict(checkpoint['optimizer_g_state_dict'])
      self.d_optimizer.load_state_dict(checkpoint['optimizer_d_state_dict'])
      self.test_progression = checkpoint['test_progression']
      self.current_epoch = checkpoint['current_epoch'] + 1
      print(f'Model loaded from {path}, starting from epoch {self.current_epoch}')
