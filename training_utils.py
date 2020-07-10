import torch

class TrainingPhase():
    """ Controls which of the two models is currently trained """
    def __init__(self, localizer, adversarial, trainingset_size, train_steps):
        self.localizer = localizer
        self.adversarial = adversarial
        self.trainingset_size = trainingset_size
        self.train_steps_localizer, self.train_steps_adversarial = train_steps

        self._train_localizer = True
        self._train_adversarial = False
    
    def update(self, epoch, batch_index):
        """ Updates which model gets trained based on the current global step """
        if not self.adversarial:
            self.train(localizer=True, adversarial=False)
        else:
            # Train the localizer and adversarial model alternatively, depending on the given train steps
            global_step = (epoch-1) * self.trainingset_size + batch_index
            if global_step % (self.train_steps_adversarial + self.train_steps_localizer) < self.train_steps_localizer:
                self.train(localizer=True, adversarial=False)
            else:
                self.train(localizer=False, adversarial=True)

    def train(self, localizer=True, adversarial=False):
        """ Set the currently trained model """
        self._train_localizer = localizer
        self._train_adversarial = adversarial

    @property
    def train_adversarial(self):
        return self._train_adversarial

    @property
    def train_localizer(self):
        return self._train_localizer

def compute_am_loss(output, targets):
    """ Computes the attention mining loss """
    one_hot_targets = torch.zeros_like(output, dtype=torch.bool)
    one_hot_targets.scatter_(1, targets.view(-1, 1), 1)
    relevant_outputs = output[one_hot_targets]
    return relevant_outputs.mean()

def compute_regularization_loss(output):
    return output.mean()

def load_optimizer(algorithm, params, learning_rate):
    if algorithm == 'sgd':
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    if algorithm == 'adam':
        return torch.optim.Adam(params, lr=learning_rate, betas=(0.5, 0.999))
    if algorithm == 'rmsprop':
        return torch.optim.RMSprop(params, lr=learning_rate)
    raise ValueError(f'Invalid optimizer algorithm specified: {algorithm}')
