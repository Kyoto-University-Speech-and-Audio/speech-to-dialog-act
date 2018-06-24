import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class CLSTMCell(LSTMCell):
    """Long short-term memory unit (LSTM) recurrent network cell.
    The default non-peephole implementation is based on:
      http://www.bioinf.jku.at/publications/older/2604.pdf
    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
    The peephole implementation is based on:
      https://research.google.com/pubs/archive/43905.pdf
    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.
    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """

    def __init__(self, num_units, context_embedding,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 forget_bias=1.0,
                 activation=None, reuse=None, name=None):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          When restoring from CudnnLSTM-trained checkpoints, use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(CLSTMCell, self).__init__(
            num_units,
            use_peepholes=use_peepholes,
            cell_clip=cell_clip,
            initializer=initializer,
            num_proj=num_proj,
            proj_clip=proj_clip,
            num_unit_shards=None,
            num_proj_shards=None,
            forget_bias=forget_bias,
            state_is_tuple=True,
            activation=activation,
            reuse=reuse,
            name=name)

        self._context_embedding = context_embedding

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        context_depth = self._context_embedding.get_shape().as_list()[1]
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (
            tf.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)
        print("hello", [input_depth + context_depth + h_depth, 4 * self._num_units])
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + context_depth + h_depth, 4 * self._num_units],
            initializer=self._initializer,
            partitioner=maybe_partitioner)
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=tf.zeros_initializer(dtype=self.dtype))
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                               initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                tf.fixed_size_partitioner(self._num_proj_shards)
                if self._num_proj_shards is not None
                else None)
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)

        self.built = True

    def call(self, inputs, state):
        """Run one step of LSTM.
        Args:
          inputs: input Tensor, 2D, `[batch, num_units].
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
        Returns:
          A tuple containing:
          - A `2-D, [batch, output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.
        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        sigmoid = tf.sigmoid

        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = tf.matmul(
            tf.concat([inputs, m_prev, self._context_embedding], 1), self._kernel)
        lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

        i, j, f, o = tf.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            m = tf.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = tf.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     tf.concat([c, m], 1))
        return m, new_state
