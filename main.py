import argparse
from train import Trainer
from test import Tester

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--isTraining', type=str, required=True, help='True for training, False for testing')

    # Train options
    parser.add_argument('--dataset_path', type=str,
                        default='D:\PycharmProjects\Text-to-Image-Synthesis-master/birds.hdf5',
                        help='hdf5 file for train')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory to save the model')
    parser.add_argument('--output_dir', type=str, default='./results')

    # Testing options
    parser.add_argument('--model_dir', type=str,
                        default='D:\PycharmProjects\Generative Adversarial Text to Image Synthesis\checkpoints/',
                        help='pre-trained model directory for training')

    # training options
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3, help='Number of channels in the training images')
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector (i.e. size of generator input)')
    parser.add_argument('--nemb', type=int, default=1024, help='Size of word embedding')
    parser.add_argument('--ngf', type=int, default=64, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='Size of feature maps in discriminator')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for optimizers')
    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparam for Adap optimizers')
    parser.add_argument('--save_model_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=50)
    return parser

if __name__== '__main__':
    args = get_args_parser().parse_args()
    if args.isTraining == True:
        trainer = Trainer(dataset_path=args.dataset_path,
                          checkpoint_dir=args.checkpoint_dir,
                          output_dir=args.output_dir,
                          ngpu=args.ngpu,
                          num_workers=args.num_workers,
                          batch_size=args.batch_size,
                          image_size=args.image_size,
                          nc=args.nc,
                          nz=args.nz,
                          nemb=args.nemb,
                          ngf=args.ngf,
                          ndf=args.ndf,
                          lr=args.lr,
                          num_epochs=args.num_epochs,
                          beta1=args.beta1,
                          save_model_interval=args.save_model_interval,
                          test_interval=args.test_interval)
    else:
        tester = Tester(model_dir=args.model_dir,
                        dataset_path=args.dataset_path,
                        output_dir=args.output_dir,
                        ngpu=args.ngpu,
                        num_workers=args.num_workers,
                        batch_size=1,
                        image_size=args.image_size,
                        nc=args.nc,
                        nz=args.nz,
                        nemb=args.nemb,
                        ngf=args.ngf,
                        ndf=args.ndf,
                        lr=args.lr,
                        beta1=args.beta1,
                        )
