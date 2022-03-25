import os

from callbacks.callbacks import Callback



class CheckpointCallback(Callback):
    def __init__(self, model, log_folder, ckpt_template, save_rate, blocks=['main'], verbose=False, **kwargs):
        super(CheckpointCallback, self).__init__()
        self.model = model
        self.save_rate = save_rate
        self.ckpt_template = ckpt_template
        self.ckpt_folder = os.path.join(log_folder, 'checkpoints')
        self.verbose = verbose
        self.blocks = blocks


    def on_epoch_end(self, logs=None):
        epoch = logs['epoch'].iloc[0]
        if epoch % self.save_rate == 0:
            if self.verbose:
                print('SAVING MODEL')
            for block in self.blocks:
                if block == 'main':
                    self.model.save_weights(os.path.join(self.ckpt_folder,
                                                         self.ckpt_template.format(epoch)),
                                            block='main',
                                            save_format='tf')
                else: 
                    self.model.save_weights(os.path.join(self.ckpt_folder,
                                                         block,
                                                         self.ckpt_template.format(epoch)),
                                            block=block,
                                            save_format='tf')


SUPPORTED_CHECKPOINT_CALLBACKS = {"ckpt": CheckpointCallback}
