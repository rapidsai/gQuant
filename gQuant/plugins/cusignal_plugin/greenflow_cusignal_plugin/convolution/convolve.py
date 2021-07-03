import numpy as np
import cupy as cp

from cusignal.convolution import convolve as cuconv
from scipy.signal import convolve as siconv

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)

__all__ = ['CusignalConvolveNode']

_CONV_DESC = '''Convolve two N-dimensional arrays.

Convolve `in1` and `in2`, with the output size determined by the
`mode` argument.

Returns:
convolve : array
    An N-dimensional array containing a subset of the discrete linear
    convolution of `in1` with `in2`.
'''

_CONV_MODE_DESC = '''mode : str {'full', 'valid', 'same'}, optional
A string indicating the size of the output:

    ``full``
       The output is the full discrete linear convolution
       of the inputs. (Default)
    ``valid``
       The output consists only of those elements that do not
       rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
       must be at least as large as the other in every dimension.
    ``same``
       The output is the same size as `in1`, centered
       with respect to the 'full' output.
'''

_CONV_METHOD_DESC = '''method : str {'auto', 'direct', 'fft'}, optional
A string indicating which method to use to calculate the convolution.

    ``direct``
       The convolution is determined directly from sums, the definition of
       convolution.
    ``fft``
       The Fourier Transform is used to perform the convolution by calling
       `fftconvolve`.
    ``auto``
       Automatically chooses direct or Fourier method based on an estimate
       of which is faster (default).
'''


class CusignalConvolveNode(Node):
    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        inports = {
            'in1': {port_type: [cp.ndarray, np.ndarray]},
            'in2': {port_type: [cp.ndarray, np.ndarray]}
        }
        outports = {
            'convolve': {port_type: [cp.ndarray, np.ndarray]},
        }
        return NodePorts(inports=inports, outports=outports)

    def conf_schema(self):
        mode_enum = ['full', 'valid', 'same']
        method_enum = ['direct', 'fft', 'auto']
        json = {
            'title': 'Cusignal Convolution Node',
            'type': 'object',
            'description': _CONV_DESC,
            'properties': {
                'mode':  {
                    'type': 'string',
                    'description': _CONV_MODE_DESC,
                    'enum': mode_enum,
                    'default': 'full'
                },
                'method': {
                    'type': 'string',
                    'description': _CONV_METHOD_DESC,
                    'enum': method_enum,
                    'default': 'auto'
                },
                'normalize': {
                    'type': 'boolean',
                    'description': 'Scale convolutioni by in2 (typically a '
                    'window) i.e. convolve(in1, in2) / sum(in2). '
                    'Default False.',
                    'default': False
                },
                'use_cpu': {
                    'type': 'boolean',
                    'description': 'Use CPU for computation via '
                    'scipy::signal.convolve. Default is False and runs on '
                    'GPU via cusignal.',
                    'default': False
                },
            },
        }
        return ConfSchema(json=json)

    def meta_setup(self):
        outports = {'convolve': {}}
        return MetaData(outports=outports)

    def process(self, inputs):
        mode = self.conf.get('mode', 'full')
        method = self.conf.get('method', 'auto')
        normalize = self.conf.get('normalize', False)
        use_cpu = self.conf.get('use_cpu', False)

        in1 = inputs['in1']
        in2 = inputs['in2']

        if use_cpu:
            conv = siconv(in1, in2, mode=mode, method=method)
            if normalize:
                scale = np.sum(in2)
        else:
            conv = cuconv(in1, in2, mode=mode, method=method)
            if normalize:
                scale = cp.sum(in2)

        if normalize:
            conv = conv if scale == 1 else conv / scale

        return {'convolve': conv}
