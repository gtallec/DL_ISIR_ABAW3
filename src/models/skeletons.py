import tensorflow.keras.models as tkm

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


SUPPORTED_SKELETONS = {"hencreg": HybridEncoderRegressor}
