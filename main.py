import gan as g
from data import mnist_dataloader

def main():
    lr = 0.0002
    gan = g.GAN(lr)
    gan.train(mnist_dataloader, num_epochs=100)
    #gan.load_model('models')
    #gan.visualize_images(1000)


if __name__ == "__main__":
    main()
