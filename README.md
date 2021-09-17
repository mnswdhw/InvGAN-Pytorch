# InvGAN-Pytorch
This repository implements the Invgan defense architecture in Pytorch. InvGAN acts as an initialisation for DefenseGAN and can help in State of the Art robustness of target models against adversarial attacks. <br>


For the trained generator and discriminator that we may need in the defense architectures of DefenseGAN and InvGAN, we use the checkpoints given in the official implementation of InvGAN. https://github.com/yogeshbalaji/InvGAN <br>

This contains tensorflow 1 weights, I have implemented the models in pytorch and have also implemented the conversion scripts that can be used to load and save the tensorflow 1 model's weights to Pytorch model's weights. The models have been tried and tested and work as an exact equivalent of the tensorflow 1 models.

The GAN equivalent architectures of the tensorflow 1 are in pyt_models.py. <br>
models.py implements the target model's architectures following the Defense-GAN paper's code snippets. model-a to model-f. <br>

The defense architectures are in the files defense_gan.py and invgan.py. At test time they take the trained generator and discriminator and employ the defense architectures to clean the adversarial images, these cleaned images are then fed to the classifier for evaluation. <br>


### InvGAN <br>

* Code follows the paper https://arxiv.org/abs/1911.10291
* Accuracy reported by the above method is 96.26% on clean images (MNIST) and 82.06 % on adversarial images (epsilon = 0.3, MNIST) 
* The Accuracy on adversarial images is more than what is reported by the paper (78%) in the same setting, this may arise due to the differences in the implementation and the framework used. <br>

![Alt Text](/assets/invgan.png)

### defenseGAN <br>

* Code follows the paper https://arxiv.org/abs/1805.06605
* Accuracy reported by the above method is 86% on clean images (MNIST) and 82 % on adversarial images (epsilon = 0.3, MNIST) <br>

![Alt Text](/assets/defense_gan.png)





