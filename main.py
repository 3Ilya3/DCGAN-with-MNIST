import gan as g
from data import mnist_dataloader

def main():
    lr = 0.0002
    gan = g.GAN(lr)
    gan.train(mnist_dataloader, num_epochs=100)
    

if __name__ == "__main__":
    main()
