import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class SPP3D(Layer):
    """Spatial pyramid pooling layer for 3D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. channels_firste lengchannels_first of channels_firste list is channels_firste number of pooling regions,
            each int in channels_firste list is channels_firste number of regions in channels_firstat pool. For example [1,2,3] would be 3
            regions wichannels_first 1, 2x2x2 and 3x3x3 max pools, so 21 outputs per feature map
    # Input shape
        5D tensor wichannels_first shape:
        `(samples, channels, rows, cols, hight)` if dim_ordering='channels_first'
        or 5D tensor wichannels_first shape:
        `(samples, rows, cols, hight, channels)` if dim_ordering='channels_last'.
    # Output shape
        2D tensor wichannels_first shape:
        `(samples, channels * sum([1*1*i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()

        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([1*1*i for i in pool_list])

        super(SPP3D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[4]

        print(self.nb_channels)

    # def compute_output_shape(self, input_shape):
    #     return input_shape #(input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SPP3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        input_shape = K.shape(x)

        if self.dim_ordering == 'channels_first':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
            num_hight = input_shape[4]
        elif self.dim_ordering == 'channels_last':
            num_rows = input_shape[1]   #8
            num_cols = input_shape[2]   #16
            num_hight = input_shape[3]  #24

        row_lengchannels_first = [K.cast(num_rows, 'float32') / i for i in self.pool_list]     #[8,4,2]
        col_lengchannels_first = [K.cast(num_cols, 'float32') / i for i in self.pool_list]     #[16,8,4]
        hight_lengchannels_first = [K.cast(num_hight, 'float32') / i for i in self.pool_list]  #[24,12,6]

        outputs = []

        if self.dim_ordering == 'channels_first':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                ix = 1
                jy = 1

                for kz in range(num_pool_regions):
                    x1 = K.cast(K.round(ix * row_lengchannels_first[pool_num]), 'int32')
                    x2 = K.cast(K.round(ix * row_lengchannels_first[pool_num] + row_lengchannels_first[pool_num]), 'int32')
                    y1 = K.cast(K.round(jy * col_lengchannels_first[pool_num]), 'int32')
                    y2 = K.cast(K.round(jy * col_lengchannels_first[pool_num] + col_lengchannels_first[pool_num]), 'int32')
                    z1 = K.cast(K.round(kz * hight_lengchannels_first[pool_num]), 'int32')
                    z2 = K.cast(K.round(kz * hight_lengchannels_first[pool_num] + hight_lengchannels_first[pool_num]), 'int32')

                    new_shape = [input_shape[0], input_shape[1],x2 - x1, y2 - y1, z2 - z1]

                    x_crop = x[:, :, x1:x2, y1:y2, z1:z2]
                    # xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(x_crop, axis=(4), keepdims=True)
                    outputs.append(pooled_val)

        elif self.dim_ordering == 'channels_last':
            for pool_num, num_pool_regions in enumerate(self.pool_list):   #[0:1 1:2 2:4]
                #####################
                # outputs = MaxPooling3D(pool_size=(row_lengchannels_first[pool_num],col_lengchannels_first[pool_num],hight_lengchannels_first[pool_num]))(x)
                # outputs = Flatten()(outputs)
                #####################
                ix = 1
                jy = 1

                for kz in range(num_pool_regions): #[1,2,4]
                    x1 = K.cast(K.round(ix * row_lengchannels_first[pool_num]), 'int32')
                    x2 = K.cast(K.round(ix * row_lengchannels_first[pool_num] + row_lengchannels_first[pool_num]), 'int32')
                    y1 = K.cast(K.round(jy * col_lengchannels_first[pool_num]), 'int32')
                    y2 = K.cast(K.round(jy * col_lengchannels_first[pool_num] + col_lengchannels_first[pool_num]), 'int32')
                    z1 = K.cast(K.round(kz * hight_lengchannels_first[pool_num]), 'int32')
                    z2 = K.cast(K.round(kz * hight_lengchannels_first[pool_num] + hight_lengchannels_first[pool_num]), 'int32')

                    new_shape = [input_shape[0], x2 - x1, y2 - y1, z2 - z1, input_shape[4]]

                    x_crop = x[:, x1:x2, y1:y2, z1:z2, :]
                    # xm = K.reshape(x_crop, new_shape)

                    pooled_val = K.max(x_crop, axis=(3), keepdims=True)
                    outputs.append(pooled_val)

        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            outputs = K.concatenate(outputs)

        return outputs
