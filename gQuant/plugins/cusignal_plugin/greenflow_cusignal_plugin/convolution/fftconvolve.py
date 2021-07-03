import numpy as np
import cupy as cp

from cusignal.convolution import fftconvolve as cufftconv
from scipy.signal import fftconvolve as sifftconv

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)

__all__ = ['CusignalFFTConvolveNode']

_FFTCONV_DESC = '''Convolve two N-dimensional arrays using FFT.

Convolve `in1` and `in2` using the fast Fourier transform method, with
the output size determined by the `mode` argument.

This is generally much faster than `convolve` for large arrays (n > ~500),
but can be slower when only a few output values are needed, and can only
output float arrays (int or object array inputs will be cast to float).

As of v0.19, `convolve` automatically chooses this method or the direct
method based on an estimation of which is faster.

Returns:
out : array
    An N-dimensional array containing a subset of the discrete linear
    convolution of `in1` with `in2`.
'''

_FFTCONV_MODE_DESC = '''mode : str {'full', 'valid', 'same'}, optional
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
   axis : tuple, optional
'''

_FFTCONV_AXES_DESC = '''axes : int or array_like of ints or None, optional
Axes over which to compute the convolution.
The default is over all axes.
'''


class CusignalFFTConvolveNode(Node):

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        inports = {
            'in1': {port_type: [cp.ndarray, np.ndarray]},
            'in2': {port_type: [cp.ndarray, np.ndarray]}
        }
        outports = {
            'fftconvolve': {port_type: [cp.ndarray, np.ndarray]},
        }
        return NodePorts(inports=inports, outports=outports)

    def conf_schema(self):
        mode_enum = ['full', 'valid', 'same']
        json = {
            'title': 'Cusignal Convolution Node',
            'type': 'object',
            'description': _FFTCONV_DESC,
            'properties': {
                'mode': {
                    'type': 'string',
                    'description': _FFTCONV_MODE_DESC,
                    'enum': mode_enum,
                    'default': 'full'
                },
                'axes': {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    'description': _FFTCONV_AXES_DESC,
                },
                'use_cpu': {
                    'type': 'boolean',
                    'description': 'Use CPU for computation via '
                    'scipy::signal.fftconvolve. Default is False and runs on '
                    'GPU via cusignal.',
                    'default': False
                },
            },
        }
        return ConfSchema(json=json)

    def meta_setup(self):
        outports = {'fftconvolve': {}}
        return MetaData(outports=outports)

    def process(self, inputs):
        mode = self.conf.get('mode', 'full')
        axes = self.conf.get('axes', [])
        use_cpu = self.conf.get('use_cpu', False)

        in1 = inputs['in1']
        in2 = inputs['in2']

        if len(axes) == 0:
            axes=None
        elif len(axes) == 1:
            axes = axes[0]

        if use_cpu:
            fftconv = sifftconv(in1, in2, mode=mode, axes=axes)
        else:
            # cp.cuda.set_allocator(None)
            cache = cp.fft.config.get_plan_cache()
            cache.clear()
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            if cache.get_size() > 0:
                cache.set_size(0)

            # if cache.get_memsize() != 0:
            #     cache.set_memsize(0)

            fftconv = cufftconv(in1, in2, mode=mode, axes=axes)

        return {'fftconvolve': fftconv}
