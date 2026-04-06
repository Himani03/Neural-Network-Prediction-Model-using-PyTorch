
import matplotlib.pyplot as plt

def plot_training(train_losses, val_losses):

    plt.figure(figsize=(10,5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("plots/training_loss.png")

    plt.show()
