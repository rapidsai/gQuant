import numpy as np
import cupy as cp

from cusignal.convolution import convolve2d as cuconv2d
from scipy.signal import convolve2d as siconv2d

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)

__all__ = ['CusignalConvolve2dNode']

_CONV2_DESC = '''Convolve two 2-dimensional arrays.
Convolve `in1` and `in2` with output size determined by `mode`, and
boundary conditions determined by `boundary` and `fillvalue`.

Returns:
out : ndarray
    A 2-dimensional array containing a subset of the discrete linear
    convolution of `in1` with `in2`.
'''

_CONV2_MODE_DESC = '''mode : str {'full', 'valid', 'same'}, optional

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

_CONV2_BOUNDARY_DESC = '''boundary : str {'fill', 'wrap', 'symm'}, optional

A flag indicating how to handle boundaries:
``fill``
   pad input arrays with fillvalue. (default)
``wrap``
   circular boundary conditions.
``symm``
   symmetrical boundary conditions.
'''

_CONV2_FILLVAL_DESC = '''fillvalue : scalar, optional
Value to fill pad input arrays with. Default is 0.
'''


class CusignalConvolve2dNode(Node):

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        inports = {
            'in1': {port_type: [cp.ndarray, np.ndarray]},
            'in2': {port_type: [cp.ndarray, np.ndarray]}
        }
        outports = {
            'convolve2d': {port_type: [cp.ndarray, np.ndarray]},
        }
        return NodePorts(inports=inports, outports=outports)

    def conf_schema(self):
        mode_enum = ['full', 'valid', 'same']
        boundary_enum = ['fill', 'wrap', 'symm']
        json = {
            'title': 'Cusignal Convolution2D Node',
            'type': 'object',
            'description': _CONV2_DESC,
            'properties': {
                'mode': {
                    'type': 'string',
                    'description': _CONV2_MODE_DESC,
                    'enum': mode_enum,
                    'default': 'full'
                },
                'boundary': {
                    'type': 'string',
                    'description': _CONV2_BOUNDARY_DESC,
                    'enum': boundary_enum,
                    'default': 'fill'
                },
                'fillvalue': {
                    'type': 'number',
                    'description': _CONV2_FILLVAL_DESC,
                    'default': 0
                },
                'use_cpu': {
                    'type': 'boolean',
                    'description': 'Use CPU for computation via '
                    'scipy::signal.convolve2d. Default is False and runs on '
                    'GPU via cusignal.',
                    'default': False
                },
            },
        }
        return ConfSchema(json=json)

    def meta_setup(self):
        outports = {'convolve2d': {}}
        return MetaData(outports=outports)

    def process(self, inputs):
        mode = self.conf.get('mode', 'full')
        boundary = self.conf.get('boundary', 'fill')
        fillvalue = self.conf.get('fillvalue', 0)
        use_cpu = self.conf.get('use_cpu', False)

        in1 = inputs['in1']
        in2 = inputs['in2']

        if use_cpu:
            conv2d = siconv2d(
                in1, in2, mode=mode, boundary=boundary, fillvalue=fillvalue)
        else:
            conv2d = cuconv2d(
                in1, in2, mode=mode, boundary=boundary, fillvalue=fillvalue)

        return {'convolve2d': conv2d}
