from train import Text2ImageDataset
import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.utils import save_image

class Tester(object):
    def __init__(self, model_dir, dataset_path, output_dir, ngpu, num_workers, batch_size, nz, ngf, ndf):

        dataset = Text2ImageDataset(dataset_path,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    split=1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # data_iterator = iter(dataloader)

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

        def denorm(tensor):
            # std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            # mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            res = torch.clamp(tensor * std + mean, 0, 1)
            return res

        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        netG = Generator(ngpu).to(device)
        netG.load_state_dict(torch.load(model_dir+'gen_600.pth'))

        f = open("text.txt", 'w', encoding='utf-8')

        print("Starting Training Loop...")
        # For each epoch
        for i, data in enumerate(dataloader, 0):
            with torch.no_grad():
                netG.eval()
                real_img = data['right_images'].to(device)
                real_embed = data['right_embed'].to(device)
                txt = data['txt']

                noise = torch.randn(real_img.size(0), nz, device=device)
                gen_images = netG(noise, real_embed).detach().cpu()
                f.write(str(i)+str(txt)+'\n')

                output = denorm(gen_images)
                output_name = os.path.abspath(output_dir) + '/output_{0}.jpg'.format(i)
                save_image(output, str(output_name))
                print(i, txt)
        f.close()