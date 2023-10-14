import numpy as np

class MaxPool:

    def __init__(self, no_channels_in, filter_size, stride, padding, debug=False):
        assert filter_size == stride, 'Max Pooling not implemented for overlapping filters'
        assert padding == 0, 'Max Pooling not implemented for padding'

        self.no_channels_in = no_channels_in
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.debug = debug

    def set_input_shape(self, d_no, ch_no, h_in, w_in):
        self.d_no, self.ch_no, self.h_in, self.w_in = d_no, ch_no, h_in, w_in

        self.h_out = int((h_in + 2 * self.padding - self.filter_size) / self.stride + 1)
        self.w_out = int((w_in + 2 * self.padding - self.filter_size) / self.stride + 1)

        assert self.h_in%self.filter_size==0, "Max Pooling only implemented for dimension divisible by filter size"
        assert self.w_in%self.filter_size==0, "Max Pooling only implemented for dimension divisible by filter size"

    def validate_input_shape(self, input_shape):
        try:
            if self.d_no != input_shape[0]:
                self.set_input_shape(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
            assert self.ch_no == input_shape[1], f'Number of channels in input ({input_shape[1]}) does not match number of channels expected ({self.ch_no})'
            assert self.h_in == input_shape[2], f'Height of input ({input_shape[2]}) does not match height expected ({self.h_in})'
            assert self.w_in == input_shape[3], f'Width of input ({input_shape[3]}) does not match width expected ({self.w_in})'
        except:
            self.set_input_shape(input_shape[0], input_shape[1], input_shape[2], input_shape[3])

    def forward(self, input, is_training=True):
        self.validate_input_shape(input.shape)

        input_shape = input.shape
        output_shape = (input_shape[0], input_shape[1], int((input_shape[2] - self.filter_size + 2 * self.padding) / self.stride + 1), int((input_shape[3] - self.filter_size + 2 * self.padding) / self.stride + 1))

        output = np.zeros(output_shape)

        in_strides = input.strides
        view_shape = (self.d_no, self.no_channels_in, self.h_out, self.w_out, self.filter_size, self.filter_size)
        view_strides = (in_strides[0], in_strides[1], self.stride * in_strides[2], self.stride * in_strides[3], in_strides[2], in_strides[3])
        view_slices = np.lib.stride_tricks.as_strided(input, view_shape, view_strides)
        output = np.max(view_slices, axis=(4, 5))
        
        max_vals = output.repeat(self.filter_size, axis=2).repeat(self.filter_size, axis=3)

        if is_training:
            self.max_mask = np.equal(input, max_vals).astype(int)

        return output
    
    def backward(self, grad_output, learning_rate):
        grad_output = grad_output.repeat(self.filter_size, axis=2).repeat(self.filter_size, axis=3)
        grad_input = self.max_mask * grad_output
        self.max_mask = None
        return grad_input

    def get_n_params(self):
        return 0