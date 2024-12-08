import matplotlib.pyplot as plt
plt . rc ('font', family = 'serif') # Optional : Use serif fonts
import numpy as np



class loss_visualizer():
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(1,figsize=(5,4))
        
        self.losses = np.loadtxt("losses.txt", comments="#", delimiter=",").reshape(-1,4)

        self.epoches             = self.losses[:,0]
        self.training_losses     = self.losses[:,1]
        self.validation_losses   = self.losses[:,2]
        
        self.ax.grid(True)
        self.ax.set_xticks(range(0, 40, 5))  # X-ticks every 2 units
        self.ax.set_yticks(range(0, 20, 2))  # X-ticks every 2 units
        self.ax.set_xlabel("Epoches")
        self.ax.set_ylabel("Losses (AEE)")
        self.ax.set_title("Losses versus Epoches")
        
    def create_plot(self):
        self.ax.plot(self.epoches, self.training_losses, label="Training Loss", marker="^", color="darkorchid", alpha=0.8)
        self.ax.plot(self.epoches, self.validation_losses, label="Validation Loss", marker="s", color="goldenrod", alpha=0.8)
        
        self.ax.legend()
        
    def save(self,path_to_save):
        plt.savefig(path_to_save)
        
_vis = loss_visualizer()
_vis.create_plot()
_vis.save("training_losses.png")
plt.show()
