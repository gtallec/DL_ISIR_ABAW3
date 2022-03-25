import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import tensorflow as tf

from models.layers.recurrent_cells import recurrent_cell


class RecurrentRegressor(tkm.Model):
    def __init__(self, n_task, recurrent_cell_args, **kwargs):
        super(RecurrentRegressor, self).__init__(**kwargs)
        self.n_task = n_task
        self.recurrent_cell = recurrent_cell(recurrent_cell_args)

        self.recurrent_dense = []
        for i in range(n_task):
            self.recurrent_dense.append(tkl.Dense(units=1,
                                                  activation='sigmoid'))
        self.concatenate = tkl.Concatenate()

    def call(self, inputs, y=None, training=None, **kwargs):
        outputs = []
        final_output = dict()
        y_k = tf.zeros((tf.shape(inputs)[0], 1))
        input_k = tf.concat([inputs, y_k], axis=1)
        state_k = [self.recurrent_cell.get_initial_state(inputs=input_k, dtype=tf.float32)]
        for k in range(self.n_task):
            (cell_output, state_k) = self.recurrent_cell(inputs=input_k, states=state_k, training=training)
            y_k = self.recurrent_dense[k](cell_output)
            outputs.append(y_k)
            if training:
                y_k = tf.expand_dims(y[:, k], -1)
            input_k = tf.concat([inputs, y_k], axis=1)           
        final_output['task_wise'] = self.concatenate(outputs)
        return final_output

class ImageToSequenceRNN(tkm.Model):
    def __init__(self, n_task, recurrent_cell_args, N_sample, **kwargs):
        super(ImageToSequenceRNN, self).__init__(**kwargs)
        self.N_sample = N_sample
        self.n_task = n_task
        self.input_encoding = tkl.Dense(units=recurrent_cell_args['units'],
                                        activation='linear')

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.recurrent_dense = tkl.Dense(units=1,
                                         activation='linear')

    def call(self, inputs, training=None, y=None, **kwargs):
        output_dict = dict()
        outputs_k = []
        batch_size = tf.shape(inputs)[0]
        if training:
            input_k = tf.zeros((batch_size, 1))
            state_k = [self.input_encoding(inputs)]
            for k in range(self.n_task):
                (cell_output, state_k) = self.recurrent_cell(inputs=input_k,
                                                             states=state_k, 
                                                             training=training)
                output_k = self.recurrent_dense(cell_output)
                outputs_k.append(output_k)
                input_k = tf.expand_dims(y[:, k], -1)
        else:
            input_k = tf.zeros((batch_size, self.N_sample, 1))
            state_k = [tf.tile(tf.expand_dims(self.input_encoding(inputs), axis=1),
                               multiples=[1, self.N_sample, 1])]

            for k in range(self.n_task):
                (cell_output, state_k) = self.recurrent_cell(inputs=input_k,
                                                             states=state_k,
                                                             training=training)
                output_k = self.recurrent_dense(cell_output)
                p_output_k = tf.math.sigmoid(output_k)

                outputs_k.append(tf.math.reduce_mean(p_output_k, axis=1))
                uniform_sampling = tf.random.uniform(shape=[batch_size, self.N_sample, 1],
                                                     minval=0,
                                                     maxval=1)
                input_k = tf.dtypes.cast((p_output_k - uniform_sampling > 0), dtype=tf.float32)

        output_dict['output'] = tf.concat(outputs_k, axis=-1)
        return output_dict

class ImageToSequenceRNNv2(tkm.Model):
    def __init__(self, n_task, recurrent_cell_args, N_sample, **kwargs):
        super(ImageToSequenceRNNv2, self).__init__(**kwargs)
        self.N_sample = N_sample
        self.n_task = n_task
        self.input_encoding = tkl.Dense(units=recurrent_cell_args['units'],
                                        activation='linear')
        self.label_encoding = tkl.Dense(units=recurrent_cell_args['units'],
                                        activation='linear')

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.recurrent_dense = tkl.Dense(units=1,
                                         activation='linear')

    def call(self, inputs, training=None, y=None, **kwargs):
        output_dict = dict()
        outputs_k = []
        batch_size = tf.shape(inputs)[0]
        if training:
            input_k = tf.zeros((batch_size, self.n_task))
            encoded_input_k = self.label_encoding(input_k)
            
            state_k = [self.input_encoding(inputs)]
            for k in range(self.n_task):
                (cell_output, state_k) = self.recurrent_cell(inputs=encoded_input_k,
                                                             states=state_k, 
                                                             training=training)
                output_k = self.recurrent_dense(cell_output)
                output_dict['timestep_{}'.format(k)] = output_k
                outputs_k.append(output_k)
                y_k = 2 * tf.tile(tf.expand_dims(y[:, k], axis=-1), multiples=[1, self.n_task]) - 1
                input_k = input_k + y_k * tf.tile(tf.expand_dims(tf.eye(self.n_task)[:, k], axis=0),
                                                  multiples=[batch_size, 1])
                encoded_input_k = self.label_encoding(input_k)
        else:
            input_k = tf.zeros((batch_size, self.N_sample, self.n_task))
            encoded_input_k = self.label_encoding(input_k) 

            state_k = [tf.tile(tf.expand_dims(self.input_encoding(inputs), axis=1),
                               multiples=[1, self.N_sample, 1])]

            for k in range(self.n_task):
                (cell_output, state_k) = self.recurrent_cell(inputs=encoded_input_k,
                                                             states=state_k,
                                                             training=training)
                output_k = self.recurrent_dense(cell_output)
                output_dict['timestep_{}'.format(k)] = tf.math.reduce_mean(output_k, axis=1)
                p_output_k = tf.math.sigmoid(output_k)

                outputs_k.append(tf.math.reduce_mean(p_output_k, axis=1))
                uniform_sampling = tf.random.uniform(shape=[batch_size, self.N_sample, 1],
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast((p_output_k - uniform_sampling > 0), dtype=tf.float32)
                y_k = 2 * tf.tile(y_k, multiples=[1, 1, self.n_task]) - 1

                input_k = input_k + y_k * tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(self.n_task)[:, k],
                                                                                axis=0),
                                                                 axis=0),
                                                  multiples=[batch_size, self.N_sample, 1])
                encoded_input_k = self.label_encoding(input_k)

        output_dict['output'] = tf.concat(outputs_k, axis=-1)
        return output_dict

class StochasticRNN(tkm.Model):
    def __init__(self, n_task, recurrent_cell_args, N_sample, previous_label_encoding=False, current_label_encoding=False, **kwargs):
        super(StochasticRNN, self).__init__(**kwargs)
        self.N_sample = N_sample
        self.n_task = n_task
        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.recurrent_dense = tkl.Dense(units=1,
                                         activation='linear')
        self.previous_label_encoding = previous_label_encoding
        self.previous_label_encoder = None
        if self.previous_label_encoding:
            self.previous_label_encoder = tkl.Dense(units=32,
                                                    activation='relu')

        self.current_label_encoding = current_label_encoding
        if self.current_label_encoding:
            self.current_label_encoder = tkl.Dense(units=32,
                                                   activation='relu')

    def call(self, inputs, training=None, y=None, **kwargs):
        output_dict = dict()
        outputs_k = []
        batch_size = tf.shape(inputs)[0]
        if training:
            parallel = []
            tiling = [1]
            y = tf.concat([tf.zeros((batch_size, 1)), y], axis=-1)
        else:
            parallel = [self.N_sample]
            tiling = [1, 1]
            inputs = tf.expand_dims(inputs, axis=1)

        inputs = tf.tile(inputs, multiples=[1, *parallel, 1])
        padded_eye = tf.concat([tf.zeros((self.n_task, 1)), tf.eye(self.n_task)], axis=-1)
        y_k = tf.zeros((batch_size, *parallel))

        for k in range(self.n_task):
            if training:
                y_k = y[:, k]

            previous_label_encoded = tf.expand_dims(y_k, axis=-1)
            current_label_encoded = tf.zeros((batch_size, *parallel, 0))

            if self.previous_label_encoding:
                previous_label = tf.tile(2 * previous_label_encoded - 1, multiples=[*tiling, self.n_task]) 
                projection = tf.reshape(padded_eye[:, k], (*tiling, self.n_task))
                projection = tf.tile(projection, multiples=[batch_size, *parallel, 1])

                previous_label = previous_label * projection
                previous_label_encoded = self.previous_label_encoder(previous_label, training=training)
                    
            if self.current_label_encoding:
                current_label = tf.reshape(padded_eye[:, k + 1], (*tiling, self.n_task))
                current_label = tf.tile(current_label, multiples=[batch_size, *parallel, 1]) 
                current_label_encoded = self.current_label_encoder(current_label, training=training)

            input_k = tf.concat([inputs, previous_label_encoded, current_label_encoded], axis=-1)

            if k == 0:
                state_0 = self.recurrent_cell.get_initial_state(inputs=input_k, dtype=tf.float32)
                if not training:
                    state_0 = tf.expand_dims(state_0, axis=1)
                state_0 = tf.tile(state_0, multiples=[1, *parallel, 1])
                state_k = [state_0]

            (cell_output, state_k) = self.recurrent_cell(inputs=input_k, states=state_k, training=training)
            output_k = self.recurrent_dense(cell_output)
 
            if not training:
                uniform_sampling = tf.random.uniform(shape=[batch_size, self.N_sample],
                                                     minval=0,
                                                     maxval=1)
                output_k = tf.math.sigmoid(output_k)
                y_k = tf.dtypes.cast((tf.squeeze(output_k, axis=-1) - uniform_sampling > 0), dtype=tf.float32)
                output_k = tf.math.reduce_mean(output_k, axis=1)

            outputs_k.append(output_k)
 
        output_dict['output'] = tf.concat(outputs_k, axis=-1)
        return output_dict


class MRNN(tkm.Model):
    def __init__(self, T, recurrent_cell_args, N_sample, **kwargs):
        super(MRNN, self).__init__()
        self.N_sample = N_sample
        self.T = T
        self.units = recurrent_cell_args['units']

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.recurrent_dense = tkl.Dense(units=1,
                                         activation='linear')
         
        label_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1.0)
        self.label_encoder = tkl.Dense(units=self.units,
                                       kernel_initializer=label_initializer,
                                       activation='linear')
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='linear')
        self.N_sample = N_sample

    def call(self, x, training=None, y=None, **kwargs):
        output_dict = dict()
        B = tf.shape(x)[0]

        if training:
            # (B, T)
            y = 2 * y - 1
            parallel = 1
        else:
            # (B, T)
            y = tf.zeros((B, self.T))
            parallel = self.N_sample

        # Initialization
        # (B, U)
        x = self.input_compresser(x)
        # (B, 1, U)
        x = tf.expand_dims(x, axis=1)
        # (B, 1, U)
        states = x
        # (B, 1)
        y_k = tf.zeros((B, 1))

        # (B, 1, T)
        y = tf.expand_dims(y, axis=1)
        logits = tf.zeros((B, 1, self.T))
        fill_mask = tf.zeros((1, self.T))

        states = tf.tile(states, [1, parallel, 1])
        y = tf.tile(y, [1, parallel, 1])
        fill_mask = tf.tile(fill_mask, [parallel, 1])
        y_k = tf.tile(y_k, [1, parallel])

        for k in range(self.T):
            # (B, parallel, T)
            y_k = tf.expand_dims(y_k, axis=-1) * tf.expand_dims(fill_mask, axis=0)
            # (B, parallel, U)
            y_k = self.label_encoder(y_k)

            # (B, parallel, U), (B, parallel, U)
            prelogits, states = self.recurrent_cell(inputs=y_k, states=states, training=training)
            # (B, parallel, 1)
            logits_k = self.recurrent_dense(prelogits)
            logits_k = tf.squeeze(logits_k, axis=-1)
            # (parallel, T)
            fill_mask = tf.tile(tf.eye(self.T)[k, :][tf.newaxis, :], [parallel, 1])
            logits = logits + tf.expand_dims(logits_k, axis=-1) * tf.expand_dims(fill_mask, axis=0) 

            if training:
                y_k = y[:, :, k]

            else:
                # (B, N_sample)
                uniform_sampling = tf.random.uniform(shape=[B, self.N_sample],
                                                     minval=0,
                                                     maxval=1)
                # (B, N_sample)

                prob_k = tf.math.sigmoid(logits_k)
                y_k = 2 * tf.dtypes.cast(prob_k - uniform_sampling > 0, dtype=tf.float32) - 1

        output_dict['loss'] = tf.math.reduce_mean(logits, axis=1)
        output_dict['global_pred'] = tf.math.reduce_mean(tf.math.sigmoid(logits), axis=1)
        return output_dict


SUPPORTED_RECURRENT_REGRESSORS = {"recurrent_regressor": RecurrentRegressor,
                                  "stochastic_rnn": StochasticRNN,
                                  "im2seq": ImageToSequenceRNN,
                                  "im2seqv2": ImageToSequenceRNNv2,
                                  "mrnn": MRNN}
