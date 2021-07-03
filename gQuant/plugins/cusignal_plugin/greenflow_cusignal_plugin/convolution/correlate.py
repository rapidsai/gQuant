import numpy as np
import cupy as cp

from cusignal.convolution import correlate as cucorr
from scipy.signal import correlate as sicorr

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)

__all__ = ['CusignalCorrelationNode']

_CORR_DESC = '''Cross-correlate two N-dimensional arrays.

Cross-correlate `in1` and `in2`, with the output size determined by the
`mode` argument.

Returns:
correlate : array
    An N-dimensional array containing a subset of the discrete linear
    cross-correlation of `in1` with `in2`.
'''

_CORR_MODE_DESC = '''The size of the output.

    ``full``
       The output is the full discrete linear cross-correlation
       of the inputs. (Default)
    ``valid``
       The output consists only of those elements that do not
       rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
       must be at least as large as the other in every dimension.
    ``same``
       The output is the same size as `in1`, centered
       with respect to the 'full' output.
'''

_CORR_METHOD_DESC = '''Method to use to calculate the correlation.

    ``direct``
       The correlation is determined directly from sums, the definition of
       correlation.
    ``fft``
       The Fast Fourier Transform is used to perform the correlation more
       quickly (only available for numerical arrays.)
    ``auto``
       Automatically chooses direct or Fourier method based on an estimate
       of which is faster (default).  See `convolve` Notes for more detail.
'''


class CusignalCorrelationNode(Node):
    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        inports = {
            'in1': {port_type: [cp.ndarray, np.ndarray]},
            'in2': {port_type: [cp.ndarray, np.ndarray]}
        }
        outports = {
            'correlate': {port_type: [cp.ndarray, np.ndarray]},
        }
        return NodePorts(inports=inports, outports=outports)

    def conf_schema(self):
        mode_enum = ['full', 'valid', 'same']
        method_enum = ['direct', 'fft', 'auto']
        json = {
            'title': 'Cusignal Correlation Node',
            'type': 'object',
            'description': _CORR_DESC,
            'properties': {
                'mode':  {
                    'type': 'string',
                    'description': _CORR_MODE_DESC,
                    'enum': mode_enum,
                    'default': 'full'
                },
                'method': {
                    'type': 'string',
                    'description': _CORR_METHOD_DESC,
                    'enum': method_enum,
                    'default': 'auto'
                },
                'scale': {
                    'type': 'number',
                    'description': 'Scale output array i.e. out / scale',
                    'default': 1
                },
                'use_cpu': {
                    'type': 'boolean',
                    'description': 'Use CPU for computation via '
                    'scipy::signal.correlate. Default is False and runs on '
                    'GPU via cusignal.',
                    'default': False
                },
            },
        }
        return ConfSchema(json=json)

    def meta_setup(self):
        outports = {'correlate': {}}
        return MetaData(outports=outports)

    def process(self, inputs):
        mode = self.conf.get('mode', 'full')
        method = self.conf.get('method', 'auto')
        scale = self.conf.get('scale', 1)
        use_cpu = self.conf.get('use_cpu', False)

        in1 = inputs['in1']
        in2 = inputs['in2']

        if use_cpu:
            corr = sicorr(in1, in2, mode=mode, method=method)
        else:
            corr = cucorr(in1, in2, mode=mode, method=method)

        corr = corr if scale == 1 else corr / scale

        return {'correlate': corr}
