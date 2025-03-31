import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from tqdm import tqdm
import os
from models import ContextUnet, ImageConditionedUnet
from utils import SpriteDataset, generate_animation, CustomImageToImageDataset
import torch.nn.functional as F



class DiffusionModel(nn.Module):
    def __init__(self, device=None, dataset_name=None, checkpoint_name=None):
        super(DiffusionModel, self).__init__()
        self.device = self.initialize_device(device)
        self.file_dir = os.path.dirname(__file__)
        self.dataset_name = self.initialize_dataset_name(self.file_dir, checkpoint_name, dataset_name)
        self.checkpoint_name = checkpoint_name
        self.nn_model = self.initialize_nn_model(self.dataset_name, checkpoint_name, self.file_dir, self.device)
        self.working_dir = "/kaggle/working/"
        self.create_dirs(self.working_dir)
        

    def train(self, batch_size=64, n_epoch=32, lr=1e-3, timesteps=500, beta1=1e-4, beta2=0.02,
              checkpoint_save_dir=None, image_save_dir=None):
        """Trains model for given inputs"""
        self.nn_model.train()        
        _ , _, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        dataset = self.instantiate_dataset(self.dataset_name, 
                            self.get_transforms(self.dataset_name), self.file_dir)
        dataloader = self.initialize_dataloader(dataset, batch_size, self.checkpoint_name, self.file_dir)
        optim = self.initialize_optimizer(self.nn_model, lr, self.checkpoint_name, self.file_dir, self.device)
        scheduler = self.initialize_scheduler(optim, self.checkpoint_name, self.file_dir, self.device)

        # for epoch in range(self.get_start_epoch(self.checkpoint_name, self.file_dir), 
                           # self.get_start_epoch(self.checkpoint_name, self.file_dir) + n_epoch):
            # ave_loss = 0

            # for x, c in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
                # x = x.to(self.device)
                # c = self.get_masked_context(c).to(self.device)
                
                # # perturb data
                # noise = torch.randn_like(x)
                # t = torch.randint(1, timesteps + 1, (x.shape[0], )).to(self.device)
                # x_pert = self.perturb_input(x, t, noise, ab_t)

                # # predict noise
                # pred_noise = self.nn_model(x_pert, t / timesteps, c=c)

                # # obtain loss
                # loss = torch.nn.functional.mse_loss(pred_noise, noise)
                
                # # update params
                # optim.zero_grad()
                # loss.backward()
                # optim.step()

                # ave_loss += loss.item()/len(dataloader)
            # scheduler.step()
            # print(f"Epoch: {epoch}, loss: {ave_loss}")
            # self.save_tensor_images(x, x_pert, self.get_x_unpert(x_pert, t, pred_noise, ab_t), 
                                    # epoch, self.working_dir, image_save_dir)
            # self.save_checkpoint(self.nn_model, optim, scheduler, epoch, ave_loss, 
                                 # timesteps, beta1, beta2, self.device, self.dataset_name,
                                 # dataloader.batch_size, self.working_dir, checkpoint_save_dir)
        count = 0
        for epoch in range(self.get_start_epoch(self.checkpoint_name, self.file_dir), 
                           self.get_start_epoch(self.checkpoint_name, self.file_dir) + n_epoch):
            ave_loss = 0

            for x, target_img in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
                x = x.to(self.device)
                target_img = target_img.to(self.device)
                
                # Optional: randomly mask target image for robustness
                #target_img = self.get_masked_target(target_img)
                target_img = self.sobel_masked_target(target_img)
                
                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0], )).to(self.device)
                x_pert = self.perturb_input(x, t, noise, ab_t)

                # predict noise using target image as condition
                pred_noise = self.nn_model(x_pert, t / timesteps, target_img)

                # obtain loss
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                
                # update params
                optim.zero_grad()
                loss.backward()
                optim.step()

                ave_loss += loss.item()/len(dataloader)
            
            scheduler.step()
            print(f"Epoch: {epoch}, loss: {ave_loss}")

            if(count%5==0):
                # Save visual progress
                self.save_tensor_images(x, x_pert, self.get_x_unpert(x_pert, t, pred_noise, ab_t),
                                        epoch, self.working_dir, image_save_dir)

                # Save checkpoint
                self.save_checkpoint(self.nn_model, optim, scheduler, epoch, ave_loss,
                                     timesteps, beta1, beta2, self.device, self.dataset_name,
                                     dataloader.batch_size, self.working_dir, checkpoint_save_dir)

            count = count+1

    @torch.no_grad()
    def sample_ddpm(self, n_samples, context=None, timesteps=None, 
                    beta1=None, beta2=None, save_rate=20, inference_transform=lambda x: (x+1)/2):
        """Returns the final denoised sample x0,
        intermediate samples xT, xT-1, ..., x1, and
        times tT, tT-1, ..., t1
        """
        if all([timesteps, beta1, beta2]):
            a_t, b_t, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        else:
            timesteps, a_t, b_t, ab_t = self.get_ddpm_params_from_checkpoint(self.file_dir,
                                                                             self.checkpoint_name, 
                                                                             self.device)
        
        self.nn_model.eval()
        samples = torch.randn(n_samples, self.nn_model.in_channels, 
                              self.nn_model.height, self.nn_model.width, 
                              device=self.device)
        intermediate_samples = [samples.detach().cpu()] # samples at T = timesteps
        t_steps = [timesteps] # keep record of time to use in animation generation
        for t in range(timesteps, 0, -1):
            print(f"Sampling timestep {t}", end="\r")
            if t % 50 == 0: print(f"Sampling timestep {t}")

            z = torch.randn_like(samples) if t > 1 else 0
            pred_noise = self.nn_model(samples, 
                                       torch.tensor([t/timesteps], device=self.device)[:, None, None, None], 
                                       context)
            samples = self.denoise_add_noise(samples, t, pred_noise, a_t, b_t, ab_t, z)
            
            if t % save_rate == 1 or t < 8:
                intermediate_samples.append(inference_transform(samples.detach().cpu()))
                t_steps.append(t-1)
        return intermediate_samples[-1], intermediate_samples, t_steps

    def perturb_input(self, x, t, noise, ab_t):
        """Perturbs given input
        i.e., Algorithm 1, step 5, argument of epsilon_theta in the article
        """
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise
    
    def instantiate_dataset(self, dataset_name, transforms, file_dir, train=True):
        """Returns instantiated dataset for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "custom_data"}, "Unknown dataset"
        
        transform, target_transform = transforms
        if dataset_name=="mnist":
            return MNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="fashion_mnist":
            return FashionMNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="sprite":
            return SpriteDataset(os.path.join(file_dir, "datasets"), transform, target_transform)
        if dataset_name=="cifar10":
            return CIFAR10(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="custom_data":
            # return CIFAR10(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
            input_dir = "/kaggle/working/data/train/source"
            output_dir = "/kaggle/working/data/train/target"
            return CustomImageToImageDataset(input_dir, output_dir, transform, target_transform)

    def get_transforms(self, dataset_name):
        """Returns transform and target-transform for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "custom_data"}, "Unknown dataset"

        if dataset_name in {"mnist", "fashion_mnist", "cifar10"}:
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: 2*(x - 0.5)
            ])
            target_transform = transforms.Compose([
                lambda x: torch.tensor([x]),
                lambda class_labels, n_classes=10: nn.functional.one_hot(class_labels, n_classes).squeeze()
            ])

        if dataset_name=="sprite":
            transform = transforms.Compose([
                transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
                lambda x: 2*x - 1       # range [-1,1]
            ])
            target_transform = lambda x: torch.from_numpy(x).to(torch.float32)
            
        if dataset_name=="custom_data":
        # For a custom image-to-image dataset
        # Input image transform
            transform = transforms.Compose([
                transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
                lambda x: 2*x - 1       # normalize to range [-1,1]
            ])
            
            # Output/target image transform
            target_transform = transforms.Compose([
                transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
                lambda x: 2*x - 1       # normalize to range [-1,1]
            ])
        
        return transform, target_transform
    
    def get_x_unpert(self, x_pert, t, pred_noise, ab_t):
        """Removes predicted noise pred_noise from perturbed image x_pert"""
        return (x_pert - (1 - ab_t[t, None, None, None]).sqrt() * pred_noise) / ab_t.sqrt()[t, None, None, None]
    
    def initialize_nn_model(self, dataset_name, checkpoint_name, file_dir, device):
        """Returns the instantiated model based on dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "custom_data"}, "Unknown dataset name"

        if dataset_name in {"mnist", "fashion_mnist"}:
            nn_model = ContextUnet(in_channels=1, height=28, width=28, n_feat=64, n_cfeat=10, n_downs=2)
        elif dataset_name=="sprite":
            nn_model = ContextUnet(in_channels=3, height=16, width=16, n_feat=64, n_cfeat=5, n_downs=2)
        elif dataset_name == "cifar10":
            nn_model = ContextUnet(in_channels=3, height=32, width=32, n_feat=64, n_cfeat=10, n_downs=4)
        elif dataset_name == "custom_data":
            nn_model = ImageConditionedUnet(
            in_channels=3,           # RGB input images
            height=256,               # image height (adjust to your image size)
            width=256,                # image width (adjust to your image size)
            n_feat=64,               # base feature channels
            n_downs=4                # number of downsampling steps (adjust based on image size)
        )
            #nn_model = ImageConditionedUnet(ContextUnet)

        if checkpoint_name:
            checkpoint = torch.load(os.path.join(self.file_dir, "checkpoints", checkpoint_name), map_location=device)
            nn_model.to(device)
            nn_model.load_state_dict(checkpoint["model_state_dict"])
            return nn_model
        return nn_model.to(device)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, 
                        timesteps, beta1, beta2, device, dataset_name, batch_size, 
                        file_dir, save_dir):
        """Saves checkpoint for given variables"""
        if save_dir is None:
            fpath = os.path.join(file_dir, "checkpoints", f"{dataset_name}_checkpoint_{epoch}.pth")
        else:
            fpath = os.path.join(save_dir, f"{dataset_name}_checkpoint_{epoch}.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "timesteps": timesteps, 
            "beta1": beta1, 
            "beta2": beta2,
            "device": device,
            "dataset_name": dataset_name,
            "batch_size": batch_size
        }
        try:
            torch.save(checkpoint, fpath)
            #torch.save(model.state_dict(), fpath)
        except RuntimeError as e:
            print(f"Failed to save checkpoint to {fpath}: {e}")
            raise

    def create_dirs(self, file_dir):
        """Creates directories required for training"""
        dir_names = ["checkpoints", "saved-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    def initialize_optimizer(self, nn_model, lr, checkpoint_name, file_dir, device):
        """Instantiates and initializes the optimizer based on checkpoint availability"""
        optim = torch.optim.Adam(nn_model.parameters(), lr=lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
        return optim

    def initialize_scheduler(self, optimizer, checkpoint_name, file_dir, device):
        """Instantiates and initializes scheduler based on checkpoint availability"""
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, 
                                                      end_factor=0.01, total_iters=50)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return scheduler
    
    def get_start_epoch(self, checkpoint_name, file_dir):
        """Returns starting epoch for training"""
        if checkpoint_name:
            start_epoch = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["epoch"] + 1
        else:
            start_epoch = 0
        return start_epoch
    
    def save_tensor_images(self, x_orig, x_noised, x_denoised, cur_epoch, file_dir, save_dir):
        """Saves given tensors as a single image"""
        if save_dir is None:
            fpath = os.path.join(file_dir, "saved-images", f"x_orig_noised_denoised_{cur_epoch}.jpeg")
        else:
            fpath = os.path.join(save_dir, f"x_orig_noised_denoised_{cur_epoch}.jpeg")
        inference_transform = lambda x: (x + 1)/2
        save_image([make_grid(inference_transform(img.detach())) for img in [x_orig, x_noised, x_denoised]], fpath)

    def get_ddpm_noise_schedule(self, timesteps, beta1, beta2, device):
        """Returns ddpm noise schedule variables, a_t, b_t, ab_t
        b_t: \beta_t
        a_t: \alpha_t
        ab_t \bar{\alpha}_t
        """
        b_t = torch.linspace(beta1, beta2, timesteps+1, device=device)
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t
    
    def get_ddpm_params_from_checkpoint(self, file_dir, checkpoint_name, device):
        """Returns scheduler variables T, a_t, ab_t, and b_t from checkpoint"""
        checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), torch.device("cpu"))
        T = checkpoint["timesteps"]
        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(T, checkpoint["beta1"], checkpoint["beta2"], device)
        return T, a_t, b_t, ab_t
    
    def denoise_add_noise(self, x, t, pred_noise, a_t, b_t, ab_t, z):
        """Removes predicted noise from x and adds gaussian noise z
        i.e., Algorithm 2, step 4 at the ddpm article
        """
        noise = b_t.sqrt()[t]*z
        denoised_x = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return denoised_x + noise
    
    def initialize_dataset_name(self, file_dir, checkpoint_name, dataset_name):
        """Initializes dataset name based on checkpoint availability"""
        if checkpoint_name:
            return torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["dataset_name"]
        return dataset_name

    def initialize_dataloader(self, dataset, batch_size, checkpoint_name, file_dir):
        """Returns dataloader based on batch-size of checkpoint if present"""
        if checkpoint_name:
            batch_size = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["batch_size"]
        return DataLoader(dataset, batch_size, True)
    
    def get_masked_context(self, context, p=0.9):
        "Randomly mask out context"
        return context*torch.bernoulli(torch.ones((context.shape[0], 1))*p)

    def get_masked_target(self, target_img, p=0.9):
        """
        Randomly mask out target image
        """
        batch_size = target_img.shape[0]
        mask = torch.bernoulli(torch.ones((batch_size, 1, 1, 1), device=target_img.device) * p)
        return target_img * mask

    def sobel_masked_target(self, target_img, p=0.9):
        """
        Randomly mask out target image while preserving edge information

        Args:
            target_img: Input target image tensor [B, C, H, W]
            p: Probability of keeping the image

        Returns:
            Processed target image with main features preserved
        """
        batch_size = target_img.shape[0]
        device = target_img.device

        # Create mask using Bernoulli distribution
        mask = torch.bernoulli(torch.ones((batch_size, 1, 1, 1), device=device) * p)

        # Create output tensor to store processed images
        processed_target = torch.zeros_like(target_img)

        for i in range(batch_size):
            img = target_img[i].clone()  # [C, H, W]

            # Sobel edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=device)

            # Compute edge map
            edge_map = torch.zeros_like(img[0])  # [H, W]
            for c in range(img.shape[0]):
                # Apply sobel filters
                padded = F.pad(img[c].unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                gx = F.conv2d(padded, sobel_x.unsqueeze(0).unsqueeze(0))
                gy = F.conv2d(padded, sobel_y.unsqueeze(0).unsqueeze(0))
                edge_strength = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
                edge_map += edge_strength

            # Normalize edge map
            if edge_map.max() > 0:
                edge_map = edge_map / edge_map.max()

            # Create edge tensor with same number of channels as original image
            edge_tensor = edge_map.unsqueeze(0).repeat(img.shape[0], 1, 1)

            # Apply mask
            if mask[i] == 1:
                # If mask is 1, keep the original image
                processed_img = img
            else:
                # If mask is 0, combine edge information with zeroed image
                processed_img = edge_tensor * img.max()  # Scale edges to original image intensity

            processed_target[i] = processed_img

        return processed_target
    
    def save_generated_samples_into_folder(self, n_samples, context, folder_path, **kwargs):
        """Save DDPM generated inputs into a specified directory"""
        samples, _, _ = self.sample_ddpm(n_samples, context, **kwargs)
        for i, sample in enumerate(samples):
            save_image(sample, os.path.join(folder_path, f"image_{i}.jpeg"))
    
    def save_dataset_test_images(self, n_samples):
        """Save dataset test images with specified number"""
        folder_path = os.path.join(self.file_dir, f"{self.dataset_name}-test-images")
        os.makedirs(folder_path, exist_ok=True)

        dataset = self.instantiate_dataset(self.dataset_name, 
                            (transforms.ToTensor(), None), self.file_dir, train=False)
        dataloader = DataLoader(dataset, 1, True)
        for i, (image, _) in enumerate(dataloader):
            if i == n_samples: break
            save_image(image, os.path.join(folder_path, f"image_{i}.jpeg"))

    def initialize_device(self, device):
        """Initializes device based on availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def get_custom_context(self, n_samples, n_classes, device):
        """Returns custom context in one-hot encoded form"""
        context = []
        for i in range(n_classes - 1):
            context.extend([i]*(n_samples//n_classes))
        context.extend([n_classes - 1]*(n_samples - len(context)))
        return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)
    
    def generate(self, n_samples, n_images_per_row, timesteps, beta1, beta2):
        """Generates x0 and intermediate samples xi via DDPM, 
        and saves as jpeg and gif files for given inputs
        """
        root = os.path.join(self.working_dir, "generated-images")
        os.makedirs(root, exist_ok=True)
        x0, intermediate_samples, t_steps = self.sample_ddpm(n_samples,
                                                             self.get_custom_context(
                                                                 n_samples, self.nn_model.n_cfeat, 
                                                                 self.device),
                                                             timesteps,
                                                             beta1,
                                                             beta2,)
        save_image(x0, os.path.join(root, f"{self.dataset_name}_ddpm_images.jpeg"), nrow=n_images_per_row)
        generate_animation(intermediate_samples,
                           t_steps, 
                           os.path.join(root, f"{self.dataset_name}_ani.gif"),
                           n_images_per_row)

