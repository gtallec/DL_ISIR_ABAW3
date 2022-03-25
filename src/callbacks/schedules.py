import tensorflow as tf

def exponential_decay(decay, value, **kwargs):
    def schedule_fun(step, val):
        return value * tf.math.pow(decay, step)
        
    return schedule_fun

def seq_transformer(d_model, batchsize, seq_len, steps_by_epoch, lin_phase=5, **kwargs):
    """ Implementation of the Linear Warmup and harmonic decay scaled with batchsize and seq_len"""
    steps_by_epoch = tf.dtypes.cast(steps_by_epoch, dtype=tf.float32)
    d_model = tf.dtypes.cast(d_model, tf.float32)
    batchsize = tf.dtypes.cast(batchsize, tf.float32)
    seq_len = tf.dtypes.cast(seq_len, tf.float32)

    def schedule_fun(step, val):
        arg1 = tf.math.pow(tf.dtypes.cast(step + 1e-6, tf.float32), -0.5)
        arg2 = step * tf.math.pow(lin_phase * steps_by_epoch, -1.5)
        vaswani_scale = 25000
        model_scale = batchsize * seq_len
        return (model_scale / vaswani_scale) * tf.math.pow(d_model, -0.5) * tf.math.minimum(arg1, arg2)
    return schedule_fun

def mt_transformer(d_model_y, batchsize, T, steps_by_epoch, lin_phase=5, **kwargs):
    """ Implementation of the Linear Warmup and harmonic decay scaled for Multi-task Decoders"""
    return seq_transformer(d_model=d_model_y,
                           batchsize=batchsize,
                           seq_len=T,
                           steps_by_epoch=steps_by_epoch,
                           lin_phase=lin_phase,
                           **kwargs)

def vit_transformer(d_model_x, batchsize, num_patches, steps_by_epoch, lin_phase=5, **kwargs):
    """ Implementation of the Linear Warmup and harmonic decay scaled for vit Encoders"""
    return seq_transformer(d_model=d_model_x,
                           batchsize=batchsize,
                           seq_len=num_patches,
                           steps_by_epoch=steps_by_epoch,
                           lin_phase=lin_phase,
                           **kwargs)


SUPPORTED_SCHEDULES = {'exponential': exponential_decay,
                       'vit_transformer': vit_transformer,
                       'mt_transformer': mt_transformer}

def schedule(schedule_args):
    schedule_type = schedule_args.pop("type")
    return SUPPORTED_SCHEDULES[schedule_type](**schedule_args)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(1e4)
    plt.plot(x, [mt_pretraining_v2(d_model_y=32,
                                   batchsize=64,
                                   steps_by_epoch=100,
                                   T=12,
                                   warmup=50,
                                   lin_phase1=10,
                                   lin_phase2=10)(i, 1) for i in x],
             label="decoder")

    plt.legend()
    plt.show()

    """
    plt.plot(x, [inception_pretraining(decay=0.96,
                                       value=1e-5,
                                       warmup_encoder=15)(i, 1) for i in x])
    """
    plt.show()
