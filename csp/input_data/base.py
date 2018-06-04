class BaseInputData():
    def __init__(self, hparams, mode):
        hparams.num_classes = self.num_classes