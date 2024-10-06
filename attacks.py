import torch
import torch.nn as nn
import torch.optim as optim


# No Attack - No modification to data
class NoAttack:
    def apply(self, model, data, target):
        # Simply return the original data and target without any attack
        return data, target


# Random Label Attack - Randomly change the labels
class RandomAttack:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def apply(self, model, data, target):
        # Replace the target labels with random labels
        random_labels = torch.randint(0, self.num_classes, target.shape)
        return data, random_labels


# Universal Adversarial Perturbation (UAP) Attack
class UAPAttack:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def apply(self, model, data, target):
        # Perturb the input data by adding a small noise (UAP)
        perturbation = self.epsilon * torch.sign(torch.randn_like(data))
        data_perturbed = data + perturbation
        data_perturbed = torch.clamp(data_perturbed, 0, 1)  # Ensure the values are in valid range
        return data_perturbed, target
class TargetedAttack:
    def __init__(self, target_label, generator):
        self.target_label = target_label
        self.generator = generator  # Pre-trained GAN generator

    def apply(self, data, labels):
        # Generate adversarial examples targeting a specific class
        adversarial_data = self.generator(data)  # Generate adversarial examples
        adversarial_labels = torch.full(labels.shape, self.target_label)  # Flip all labels to target class
        return adversarial_data, adversarial_labels


# GAN-Based Attack
class GANAttack:
    def __init__(self, generator, discriminator, latent_dim=100):
        """
        Args:
            generator: The Generator network of GAN.
            discriminator: The Discriminator network of GAN.
            latent_dim: The latent space dimensionality for GAN noise input.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        # Set optimizers for the generator and discriminator
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        # Loss function for GAN (binary cross-entropy)
        self.loss_fn = nn.BCELoss()

    def apply(self, model, data, target):
        """
        Apply the GAN attack to manipulate or reconstruct data.
        
        Args:
            data: The input data to be attacked.
        
        Returns:
            The generated adversarial data.
        """
        # Generate adversarial data using the GAN
        adversarial_data = self.generate_adversarial_data(data.size(0))
        return adversarial_data, target

    def generate_adversarial_data(self, batch_size):
        """
        Generate adversarial data by sampling noise and using the generator.
        """
        # Sample random noise (latent vector)
        z = torch.randn(batch_size, self.latent_dim)
        adversarial_data = self.generator(z)
        return adversarial_data

    def train_gan(self, real_data, num_epochs=100):
        """
        Train the GAN using real client data.
        
        Args:
            real_data: The real data to train the GAN.
            num_epochs: The number of training epochs.
        """
        batch_size = real_data.size(0)

        for epoch in range(num_epochs):
            # --- Train Discriminator ---
            # Sample random noise and generate fake data
            z = torch.randn(batch_size, self.latent_dim)
            fake_data = self.generator(z)

            # Labels for real (1) and fake (0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Discriminator output and loss for real data
            self.optimizer_D.zero_grad()
            real_output = self.discriminator(real_data)
            real_loss = self.loss_fn(real_output, real_labels)

            # Discriminator output and loss for fake data
            fake_output = self.discriminator(fake_data.detach())
            fake_loss = self.loss_fn(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_D.step()

            # --- Train Generator ---
            self.optimizer_G.zero_grad()
            fake_output = self.discriminator(fake_data)
            g_loss = self.loss_fn(fake_output, real_labels)  # Generator tries to fool the discriminator
            g_loss.backward()
            self.optimizer_G.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

        print("GAN training completed!")


# Sample Generator and Discriminator classes for the GAN attack
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim),
            nn.Tanh()  # Outputs values in the range (-1, 1)
        )

    def forward(self, z):
        return self.fc(z)


class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Outputs a probability
        )

    def forward(self, x):
        return self.fc(x)
