import argparse
from bb import BasicBaseline, FederatedBaseline
from utils.attacks import NoAttack, RandomAttack, UAPAttack, GANAttack, TargetedAttack
from utils.defense import NoDefense, FlippedLabelsDefense
from utils.attacks import GANAttack, Generator, Discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Baseline with Attacks and Defenses")
    
    # Model options
    parser.add_argument('--model', type=str, choices=['basic', 'federated'], default='basic',
                        help="Type of model: 'basic' for a single CNN, 'federated' for FL model.")
    
    # Attack options
    parser.add_argument('--attack', type=str, choices=['none', 'random', 'targeted', 'uap', 'gan'], default='none',
                        help="Attack type to apply: 'none', 'random', 'targeted', 'uap', or 'gan'.")
    
    # Defense options
    parser.add_argument('--defense', type=str, choices=['none', 'flipped'], default='none',
                        help="Defense mechanism to apply: 'none' or 'flipped'.")
    
    # Training options
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--rounds', type=int, default=3, help="Number of federated rounds (only for federated model).")
    parser.add_argument('--clients', type=int, default=5, help="Number of clients for federated learning.")
    parser.add_argument('--malicious_clients', type=int, default=0, help="Number of malicious clients in FL.")
    
    return parser.parse_args()


def configure_attack(attack_type, model):
    """
    Configure the attack method based on user input.
    
    Args:
        attack_type (str): Type of attack ('gan', 'none', 'random', etc.).
        model: The model to apply the attack to.
        
    Returns:
    
        Attack instance.
    """
    if attack_type == 'gan':
        # Initialize GAN generator and discriminator
        latent_dim = 100
        data_dim = 28 * 28  # Assuming FashionMNIST 28x28 images as input data
        generator = Generator(latent_dim=latent_dim, data_dim=data_dim)
        discriminator = Discriminator(data_dim=data_dim)
        
        # Return GAN attack instance with generator and discriminator
        return GANAttack(generator=generator, discriminator=discriminator, latent_dim=latent_dim)
    elif attack_type == 'none':
        return NoAttack()
    elif attack_type == 'random':
        return RandomAttack(num_classes=10)
    # elif attack_type == 'targeted':
    #     return TargetedAttack(target_label=1, class_label=0)  # Example attack targeting label 1 to be classified as 0
    elif attack_type == 'uap':
        return UAPAttack(epsilon=0.1)  
    elif attack_type == 'targeted':
    # Example: Attacking label '1' to be classified as '0'
       return TargetedAttack(target_label=1, class_label=0)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def configure_defense(defense_type):
    """
    Configure the defense mechanism based on user input.
    
    Args:
        defense_type (str): Type of defense ('none', 'flipped').
    
    Returns:
        Defense instance.
    """
    if defense_type == 'none':
        return NoDefense()
    elif defense_type == 'flipped':
        return FlippedLabelsDefense(num_classes=10)
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")


if __name__ == "__main__":
    args = parse_args()

    # Model configuration
    if args.model == 'basic':
        model = BasicBaseline()  # Initialize basic CNN model
        model.load_data()  # Load data into the basic model
    elif args.model == 'federated':
        model = FederatedBaseline(num_clients=args.clients)  # Initialize federated learning model
        model.load_data()  # Load data for federated learning
    
    # Configure attack and defense
    attack = configure_attack(args.attack, model)
    defense = configure_defense(args.defense)

    # Apply attack and defense to the model
    model.configure_attack(attack, args.malicious_clients)
    model.configure_defense(defense)

    # Training and evaluation
    if args.model == 'basic':
        # Training the basic model
        model.train(num_epochs=args.epochs, lr=0.001)
        model.test()  # Test the basic model
    elif args.model == 'federated':
        # Training the federated model across rounds
        model.train(num_epochs=args.epochs, rounds=args.rounds, lr=0.001)
        model.test()  # Test the federated model



# (x_train, _), (_, _) = mnist.load_data()
# x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize images to [-1, 1]
# x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension (28, 28, 1)
