import gan as g
from data import mnist_dataloader

def main():
    lr = 0.0002
    gan = g.GAN(lr, model_path='models/model_50')
    gan.train(mnist_dataloader, num_epochs=50)
    

if __name__ == "__main__":
    main()
