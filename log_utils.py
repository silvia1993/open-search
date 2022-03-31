import matplotlib

matplotlib.use('Agg')
import wandb


class TensorBoardWrapper:

    def __init__(self, args, folder):
        self.args = args
        wandb.init(project="Fairness", name=folder, config=vars(args), entity="vandal_to_paint", resume="never")

    def add_scalar(self, *args):
        name, value, step = args
        wandb.log({name: value}, step=step)






