from callbacks.callbacks import CallbackList
from callbacks.interface import get_callback

class CallbackBuilder:
    def __init__(self, experiment_parser, verbose=False):
        self.experiment_parser = experiment_parser
        self.verbose = verbose

    def build_callback_list(self):
        pre_callback_list = self.experiment_parser.get_pre_callbacks()
        callback_list = CallbackList()

        for pre_callback in pre_callback_list:
            # Adding parsed arguments
            pre_callback['model'] = self.experiment_parser.get_model()
            # pre_callback['optimizers'] = self.experiment_parser.get_optimizers()
            pre_callback['log_folder'] = self.experiment_parser.get_log_folder()
            # pre_callback['dataset_meta'] = self.experiment_parser.get_dataset_meta()
            pre_callback['verbose'] = self.verbose

            callback_instance = get_callback(pre_callback)
            callback_list.add_callback(callback_instance)

        return callback_list
