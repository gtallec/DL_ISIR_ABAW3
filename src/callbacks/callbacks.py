
class Callback(object):
    def __init__(self, **kwargs):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, logs=None):
        pass

    def on_epoch_end(self, logs=None):
        pass


class CallbackList(Callback):
    def __init__(self):
        super(CallbackList, self).__init__()
        self.callback_list = []

    def add_callback(self, callback):
        self.callback_list.append(callback)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callback_list:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callback_list:
            callback.on_batch_end(batch, logs)
    
    def on_epoch_end(self, logs=None):
        for callback in self.callback_list:
            callback.on_epoch_end(logs)

    def on_epoch_begin(self, logs=None):
        for callback in self.callback_list:
            callback.on_epoch_begin(logs)

    def on_train_begin(self, logs=None):
        for callback in self.callback_list:
            callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        for callback in self.callback_list:
            callback.on_train_end(logs=logs)



