# Conditional DDPM

## Introduction

Initially, as part of research process, I thought of checking the basic ML steps like providing inputs and training the model with the outputs using a supervised learning method, where loss is calculated and pixel data could be changed accordingly. In order to perform this, I needed data, that I first thought to create manually. But later, I thought to search and see if any similar dataset was available and that is how I got the CHRD-3K dataset. This was manually done by skilled professionals where the wrinkles were retouched and edited. Now I tried the training, but this scenario was not possible and this made me think of image-to-image generation models. 
Different type of methods like GANs and VAEs were available, my interest came towards the diffusion models. Since the latest image generation models like DALL-E uses diffusion model, I decided to use a similar model for the use case. 

Understanding the diffusion model was little tricky. Even though the brief of the model appeared to be easy, the working, loss calculation and regeneration of the image with these factors seemed to be very deep. I then started going through different videos and papers which explained the working of diffusion model. I came to understand that the images are being generated from the noise. Then I came to know about denoising diffusion model. Now my concern was to use the stable diffusion model or the denoising diffusion model. The stable diffusion makes use of the latent space and then generate the images, but in the denoising diffusion model the UNet helps in keeping the spatial information intact. In stable diffusion there are chances that additional information might get added. This point then made me think if it was possible to give some information to keep the images as the same but work on the wrinkles. This is when I came across the conditional diffusion models, which made me realize that we can use the conditional denoising diffusion model to remove the wrinkles.  

## Setting Up the Environment

These experiments were done in the Kaggle environment. GPU P100 was used as the accelerator. A total of 100 epochs were ran. They were run at several intervals of 20 and 25. Checkpoint weights are available at 15th, 20th, 41st, 62nd ,83rd and .. epochs. The maximum size of Kaggle working directory is 20 GB and a limit of 30 hours per week, and each checkpoint has a size of almost 4GB, hence, I had to run several times to get up to 100 epochs at least.

## Files Changed:
1.	Diffusion_model.py
2.	Train.py
3.	Models.py
4.	Checkpoints folder
5.	Kaggle-code
   
## Lessons Learnt:
1.	Vibe coding won’t be a good approach to run a big and complex task. 
2.	If model for a particular task exists, it would be better to implement that model as guided and then perform the code changes.  Sub tasking is better approach to resolve issues.
3.	Having a draft idea of the overall working of the proposed model will make it easy to edit the code accordingly.
4.	Getting the model trained for our particular and then performing a fine-tuning is likely to improve the performance of the model. 
5.	Still trying to derive the mathematics involved in the diffusion model. 😊
