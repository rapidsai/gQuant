import numpy as np
import cupy as cp
import ast

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)

__all__ = ['DummySignalNode']



def exec_then_eval(code):
    # https://stackoverflow.com/questions/39379331/python-exec-a-code-block-and-eval-the-last-line
    block = ast.parse(code, mode='exec')

    # assumes last node is an expression
    last = ast.Expression(block.body.pop().value)

    _globals, _locals = {}, {}
    exec(compile(block, '<string>', mode='exec'), _globals, _locals)
    return eval(compile(last, '<string>', mode='eval'), _globals, _locals)


class DummySignalNode(Node):
    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        inports = {}
        outports = {
            'out1': {port_type: [cp.ndarray, np.ndarray]},
            'out2': {port_type: [cp.ndarray, np.ndarray]},
        }
        return NodePorts(inports=inports, outports=outports)

    def conf_schema(self):
        json = {
            'title': 'Dummany Signal Node.',
            'type': 'object',
            'description': 'Inject signals into greenflow taskgraphs. Use '
            'CAUTION. Only run trusted code.',
            'properties': {
                'pycode': {
                    'type': 'string',
                    'title': 'Signal Code',
                    'description': 'Enter python code to generate signal. '
                    'The code must have a dictionary ``myout`` variable with '
                    'keys: out1 and out2. The ``myout`` must be the last '
                    'line. Keep it simple please.'
                },
            },
            # 'required': ['pycode'],
        }
        ui = {'pycode': {'ui:widget': 'textarea'}}
        return ConfSchema(json=json, ui=ui)

    def meta_setup(self):
        outports = {'out1': {}, 'out2': {}}
        return MetaData(outports=outports)

    def process(self, inputs):
        pycode = self.conf.get('pycode')
        # print('Task id: {}; Node type: {}\nPYCODE:\n{}'.format(
        #     self.uid, 'DummySignalNode', pycode))

        if pycode:
            myout = exec_then_eval(pycode)
            return myout

        # slen = int(1e8)
        # sig_noise = cp.random.rand(slen) + cp.random.randn(slen)
        # sig2 = cp.ones(128)
        # return {'out1': sig_noise, 'out2': sig2}
        raise RuntimeError('Task id: {}; Node type: {}\n'
                           'No code provided. Nothing to output.'
                           .format(self.uid, 'DummySignalNode'))
