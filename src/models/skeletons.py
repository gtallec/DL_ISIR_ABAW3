import tensorflow.keras.models as tkm
import tensorflow as tf

class MaskedEncoderRegressor(tkm.Model):
    def __init__(self, encoder, regressor, **kwargs):
        super(MaskedEncoderRegressor, self).__init__()
        self.encoder = encoder
        self.regressor = regressor

    def call(self, x, **kwargs):
        """ x of size (B, S, H, W, C) """
        padding_mask = 1 - tf.dtypes.cast(tf.math.reduce_sum(tf.math.abs(x), axis=(2, 3, 4)) == 0, tf.float32)
        encoded_x = self.encoder(x, padding_mask=padding_mask, **kwargs)
        prediction = self.regressor(encoded_x, padding_mask=padding_mask, **kwargs)
        return prediction

class EncoderRegressorModel(tkm.Model):
    def __init__(self, encoder, regressor, **kwargs):
        super(EncoderRegressorModel, self).__init__()
        self.encoder = encoder
        self.regressor = regressor

    def call(self, x, **kwargs):
        encoded_x = self.encoder(x, **kwargs)
        prediction = self.regressor(encoded_x, **kwargs)
        prediction['input'] = x
        return prediction

class HybridEncoderRegressor(tkm.Model):
    def __init__(self, encoder, transformer, decoder, **kwargs):
        super(HybridEncoderRegressor, self).__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder

    def call(self, x, **kwargs):
        encoded_x = self.encoder(x, **kwargs)
        transformed_x, blocks_xx = self.transformer(encoded_x, **kwargs)
        output_dict = self.decoder(transformed_x, **kwargs)
        
        for i in range(len(blocks_xx)):
            output_dict['layer_{}_block_xx'.format(i)] = blocks_xx[i]

        return output_dict

class MixtFromXNet(tkm.Model):
    def __init__(self, encoder, regressor, mixture, **kwargs):
        super(MixtFromXNet, self).__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.mixture = mixture

    def call(self, x, **kwargs):
        output_dict = dict()
        track_grad = dict()

        encoded_x = self.encoder(x, **kwargs)
        mixture_dict = self.mixture(x, **kwargs)
        
        # Harvest from mixture dict
        track_grad_mixture = mixture_dict.pop('track_grad', dict())
        track_grad.update(track_grad_mixture)
        output_dict.update(mixture_dict)

        pred_dict = self.regressor(inputs=encoded_x, mixture_logits=mixture_dict['logits'], **kwargs)

        # Harvest from prediction dict
        track_grad_pred = pred_dict.pop('track_grad', dict())
        track_grad.update(track_grad_pred)
        output_dict.update(pred_dict)

        output_dict['track_grad'] = track_grad
        output_dict['input'] = x
        return output_dict

class MixtFromEncodedXNet(tkm.Model):
    def __init__(self, encoder, regressor, mixture, **kwargs):
        super(MixtFromEncodedXNet, self).__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.mixture = mixture

    def call(self, x, **kwargs):
        output_dict = dict()
        track_grad = dict()

        encoded_x = self.encoder(x, **kwargs)
        mixture_dict = self.mixture(encoded_x, **kwargs)
        
        # Harvest from mixture dict
        track_grad_mixture = mixture_dict.pop('track_grad', dict())
        track_grad.update(track_grad_mixture)
        output_dict.update(mixture_dict)

        pred_dict = self.regressor(inputs=encoded_x, mixture_logits=mixture_dict['logits'], **kwargs)

        # Harvest from prediction dict
        track_grad_pred = pred_dict.pop('track_grad', dict())
        track_grad.update(track_grad_pred)
        output_dict.update(pred_dict)

        output_dict['track_grad'] = track_grad
        output_dict['input'] = x
        return output_dict

class CMixtureFromEmbedding(tkm.Model):
    def __init__(self, encoder, regressor, mixture, **kwargs):
        super(CMixtureFromEmbedding, self).__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.mixture = mixture

    def call(self, x, **kwargs):
        encoded_x = self.encoder(x, **kwargs)
        mixture_logits = self.mixture(None, **kwargs) 
        output_dict = self.regressor(inputs=encoded_x, mixture_logits=mixture_logits, **kwargs)
        return output_dict


SUPPORTED_SKELETONS = {"encoder_regressor": EncoderRegressorModel,
                       "hencreg": HybridEncoderRegressor,
                       "mencreg": MaskedEncoderRegressor,
                       "x2mixt": MixtFromXNet,
                       "encodex2mixt": MixtFromEncodedXNet,
                       "cmixture": CMixtureFromEmbedding}


if __name__ == '__main__':
    print(invert_list([0, 2, 4, 1, 3, 5]))

