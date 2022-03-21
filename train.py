import os
import io
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image


class Text2ImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, split=0):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5pyint = lambda x:int(np.array(x))
        
    def __len__(self):
        f = h5py.File(self.dataset_dir, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()
        return length
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_dir, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]
        
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        
        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())
        
        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        right_image = self.validate_image(right_image)
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))
        wrong_image = self.validate_image(wrong_image)
        txt = np.array(example['txt']).astype(str)
        
        sample = {
            'right_images' : torch.FloatTensor(right_image),
            'right_embed' : torch.FloatTensor(right_embed),
            'wrong_images' : torch.FloatTensor(wrong_image),
            'inter_embed' : torch.FloatTensor(inter_embed),
            'txt': str(txt)
            }
        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample
    
    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb
        return img.transpose(2, 0, 1)

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


class Trainer(object):
    def __init__(self, dataset_path, checkpoint_dir, output_dir, ngpu, num_workers, batch_size, image_size,
                 nc, nz, nemb, ngf, ndf, lr, num_epochs, beta1, save_model_interval, test_interval, npics=4):

        dataset = Text2ImageDataset(dataset_path,
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        data_iterator = iter(dataloader)
        sample = next(data_iterator)

        fig = plt.figure()

        def multiple_imshow(fig, sample, idx):
            out = sample['right_images'][idx]
            out = out.data.mul_(127.5).add_(127.5).permute(1, 2, 0).byte().cpu().numpy()
            ax = fig.add_subplot(1, npics, idx+1)
            plt.imshow(out)

        final_title = ''
        for i in range(npics):
            multiple_imshow(fig, sample, i)
            final_title += str(sample['txt'][i]) + '\n'

        plt.show()

        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.ngf = 64
                self.nc = 3
                self.nemb = 1024
                self.nproj = 128
                self.nz = 100
                self.ninput = self.nz + self.nproj

                self.projection = nn.Sequential(
                    nn.Linear(self.nemb, self.nproj),
                    nn.BatchNorm1d(self.nproj),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(self.ninput, self.ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(self.ngf * 8),
                    nn.ReLU(True),
                    # state size. (nfg*8) x 4 x 4
                    nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ngf * 4),
                    nn.ReLU(True),
                    # state size. (nfg*4) x 8 x 8
                    nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ngf * 2),
                    nn.ReLU(True),
                    # state size. (nfg*2) x 16 x 16
                    nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # state size. (nfg) x 32 x 32
                    nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )

            def forward(self, z, emb):
                projected_emb = self.projection(emb)
                latent_vector = torch.cat([z, projected_emb], 1)
                latent_vector = latent_vector.unsqueeze(2).unsqueeze(3)
                output = self.main(latent_vector)
                return output

        class Discriminator(nn.Module):
            def __init__(self, ngpu):
                super(Discriminator, self).__init__()
                self.ngpu = ngpu
                self.ndf = 64
                self.nc = 3
                self.nemb = 1024
                self.nproj = 128
                self.ninput = self.nemb + self.nproj
                self.main1 = nn.Sequential(
                    # input is (nc) x 64 x 64
                    nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x 32 x 32
                    nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ndf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
                    nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
                )
                self.projection = nn.Sequential(
                    nn.Linear(self.nemb, self.nproj),
                    nn.BatchNorm1d(self.nproj),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
                self.main2 = nn.Sequential(
                    nn.Conv2d(ndf * 8 + self.nproj, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, img, emb):
                out = self.main1(img)
                emb = self.projection(emb)
                # [8,128] -> [8,128, 4, 4]
                replicated_emb = emb.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
                out = torch.cat([out, replicated_emb], 1)
                out = self.main2(out)
                return out

        def save_checkpoint(netD, netG, dir_path, epoch):
            path = os.path.abspath(dir_path)
            if not os.path.exists(path):
                os.makedirs(path)

            torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
            torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

        def smooth_label(tensor, offset):
            return tensor + offset

        def denorm(tensor):
            # std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            # mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            res = torch.clamp(tensor * std + mean, 0, 1)
            return res


        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # custom weights initialization called on netG and netD
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        netG = Generator(ngpu).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weight_init function to randomly initialize all weights
        # to mean=0, stdev=0.02
        netG.apply(weights_init)

        print(netG)

        netD = Discriminator(ngpu).to(device)

        # Handle nulti-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        # to mean=0, stdev=0.2.
        netD.apply(weights_init)

        print(netD)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        # fixed_noise = torch.randn(batch_size, nz, image_size, image_size, device=device)
        # fixed_embed = torch.randn(batch_size, nemb, device=device)

        # Establish convention for real and fake labels during trainig
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        sample = next(data_iterator)
        print(sample['right_images'][0].shape)
        print(sample['right_embed'][0].shape)
        print(len(sample['txt'][0]))

        # Training
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        # iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ####################
                # (1) Update D network: maximize log(D(x)) + log(1-D(G(z)))
                ####################
                netD.zero_grad()
                real_img = data['right_images'].to(device)
                real_embed = data['right_embed'].to(device)
                wrong_img = data['wrong_images'].to(device)
                inter_embed = data['inter_embed'].to(device)

                label = torch.full((real_img.size(0),), real_label, dtype=torch.float, device=device)
                smoothed_real_labels = torch.FloatTensor(smooth_label(label.cpu().numpy(), -0.1)).cuda()
                # train with {real image, right text} -> real
                output = netD(real_img, real_embed).view(-1)
                score_r = criterion(output, smoothed_real_labels)
                score_r.backward()
                D_r = output.mean().item()

                # train with {real image, wrong text} -> fake
                label.fill_(fake_label)
                beta_INT = 0.5
                output = netD(real_img, real_embed*beta_INT+inter_embed*(1-beta_INT)).view(-1)
                score_w = criterion(output, label)
                score_w.backward()
                D_w = output.mean().item()

                # train with {fake image, right text} -> fake
                noise = torch.randn(real_img.size(0), nz, device=device)
                fake_images = netG(noise, real_embed)
                output = netD(fake_images, real_embed).view(-1)
                score_f = criterion(output, label)
                score_f.backward()
                D_f = output.mean().item()
                errD = score_r + score_w/2 + score_f/2
                optimizerD.step()

                # # Train with all-fake batch
                # # Generate batch of latent vectors
                # noise = torch.randn(real_img.size(0), nz, device=device)
                # wrong_embed = torch.rand(real_img.size(0), nemb, device=device)
                # # Generate batch of latent vectors
                # fake = netG(noise, real_embed)
                #
                # # classify all fake batch with D
                # output = netD(fake.detach(), real_embed).view(-1)
                # errD_fake = criterion(output, label)
                # # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                # errD_fake.backward()
                # D_G_z1 = output.mean().item()
                # errD = errD_real + errD_fake

                # Update D

                ####################
                # (2) Update G network: maximize log(D(g(z)))
                ####################
                netG.zero_grad()
                label.fill_(real_label)

                noise = torch.randn(real_img.size(0), nz, device=device)
                fake_images = netG(noise, real_embed)
                output = netD(fake_images, real_embed).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                if i % test_interval == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(r): %.4f\tD(w): %.4f\tD(f): %.4f'
                          # 'D(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_r, D_w, D_f))

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if i % test_interval == 0 or i == len(dataloader)-1:
                    with torch.no_grad():
                        temp_noise = torch.randn(real_img.size(0), nz, device=device)
                        fake = netG(temp_noise, real_embed).detach().cpu()
                        # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                        output = denorm(fake.cpu())
                        output_name = os.path.abspath(output_dir) + '/output_epoch_{:0}.jpg'.format(epoch)
                        save_image(output, str(output_name))
                        # fig = plt.figure(figsize=(8, 8))
                        # out = np.transpose(img_list[-1], (1, 2, 0))
                        # plt.imshow(out)
                        # plt.show()
                        # plt.savefig('./results/gen_images_{0}.png'.format(epoch))

                if (epoch+1) % save_model_interval == 0 or epoch == num_epochs-1:
                    save_checkpoint(netD, netG, checkpoint_dir, epoch+1)

                # iters += 1

            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            # plt.show()
            fig_name = os.path.abspath(output_dir) + '/plot_epoch_{:0}.jpg'.format(epoch)
            plt.savefig(fig_name)
