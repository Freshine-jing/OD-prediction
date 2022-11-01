from multiHeadAttention import *

class PositionWiseFeedForward(Layer):

    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim

class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class Transformer(Layer):

    def __init__(self, vocab_size, model_dim,
                 n_heads=8, encoder_stack=6, decoder_stack=6, feed_forward_size=2048, dropout_rate=0.1, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="embeddings")
        super(Transformer, self).build(input_shape)

    def encoder(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')

        masks = K.equal(inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5  # Scale
        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            attention_input = [encodings, encodings, encodings, masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += encodings
            attention_out = LayerNormalization()(attention_out)
            # Feed-Forward
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

        return encodings, masks

    def decoder(self, inputs):
        decoder_inputs, encoder_encodings, encoder_masks = inputs
        if K.dtype(decoder_inputs) != 'int32':
            decoder_inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, decoder_inputs)
        embeddings *= self._model_dim ** 0.5  # Scale
        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = LayerNormalization()(masked_attention_out)

            # Multi-head-Attention
            attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            attention_input = [masked_attention_out, encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += masked_attention_out
            attention_out = LayerNormalization()(attention_out)

            # Feed-Forward
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

        # Pre-Softmax 与 Embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return outputs

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)
        encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
        return encoder_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self._vocab_size)