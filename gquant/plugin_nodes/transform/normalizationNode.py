from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   PortsSpecSchema, NodePorts)
from .data_obj import NormalizationData
from collections import OrderedDict


class NormalizationNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'df_in'
        self.OUTPUT_PORT_NAME = 'df_out'
        self.INPUT_NORM_MODEL_NAME = 'norm_data_in'
        self.OUTPUT_NORM_MODEL_NAME = 'norm_data_out'
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required,
            self.INPUT_NORM_MODEL_NAME: cols_required
        }

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            },
            self.INPUT_NORM_MODEL_NAME: {
                port_type: NormalizationData
            }
        }

        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: types
            },
            self.OUTPUT_NORM_MODEL_NAME: {
                port_type: NormalizationData
            }
        }

        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            input_ports.update({self.INPUT_PORT_NAME:
                                {port_type: determined_type}})
            output_ports.update({self.OUTPUT_PORT_NAME: {
                                 port_type: determined_type}})
            # connected
            return NodePorts(inports=input_ports,
                             outports=output_ports)
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self):
        self.required = {
            self.INPUT_PORT_NAME: {},
            self.INPUT_NORM_MODEL_NAME: {}
        }
        if 'columns' in self.conf and self.conf.get('include', True):
            cols_required = {}
            for col in self.conf['columns']:
                cols_required[col] = None
            self.required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_NORM_MODEL_NAME: cols_required
            }
        output_cols = {
            self.OUTPUT_PORT_NAME: self.required[self.INPUT_PORT_NAME],
            self.OUTPUT_NORM_MODEL_NAME: self.required[
                self.INPUT_NORM_MODEL_NAME]
        }
        input_columns = self.get_input_columns()
        if (self.INPUT_NORM_MODEL_NAME in input_columns and
                self.INPUT_PORT_NAME in input_columns):
            cols_required = input_columns[self.INPUT_NORM_MODEL_NAME]
            self.required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_NORM_MODEL_NAME: cols_required
            }
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
                self.OUTPUT_NORM_MODEL_NAME: cols_required
            }
            return output_cols
        elif (self.INPUT_NORM_MODEL_NAME in input_columns and
              self.INPUT_PORT_NAME not in input_columns):
            cols_required = input_columns[self.INPUT_NORM_MODEL_NAME]
            self.required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_NORM_MODEL_NAME: cols_required
            }
            output_cols = {
                self.OUTPUT_PORT_NAME: cols_required,
                self.OUTPUT_NORM_MODEL_NAME: cols_required
            }
            return output_cols
        elif (self.INPUT_NORM_MODEL_NAME not in input_columns and
              self.INPUT_PORT_NAME in input_columns):
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                else:
                    included_colums = [col for col in enums
                                       if col not in self.conf['columns']]
                cols_required = OrderedDict()
                for col in included_colums:
                    if col in col_from_inport:
                        cols_required[col] = col_from_inport[col]
                    else:
                        cols_required[col] = None
                self.required = {
                    self.INPUT_PORT_NAME: cols_required,
                    self.INPUT_NORM_MODEL_NAME: cols_required
                }
                output_cols = {
                    self.OUTPUT_PORT_NAME: col_from_inport,
                    self.OUTPUT_NORM_MODEL_NAME: cols_required
                }
            return output_cols
        return output_cols

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Normalization Node configure",
            "type": "object",
            "description": "Normalize the columns to have zero mean and std 1",
            "properties": {
                "columns":  {
                    "type": "array",
                    "description": """an array of columns that need to
                     be normalized, or excluded from normalization depending
                     on the `incldue` flag state""",
                    "items": {
                        "type": "string"
                    }
                },
                "include":  {
                    "type": "boolean",
                    "description": """if set true, the `columns` need to be 
                    normalized. if false, all dataframe columns except the 
                    `columns` need to be normalized""",
                    "default": True
                },
            },
            "required": [],
        }
        ui = {}
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        normalize the data to zero mean, std 1

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        if self.INPUT_NORM_MODEL_NAME in inputs:
            norm_data = inputs[self.INPUT_NORM_MODEL_NAME].data
            input_columns = self.get_input_columns()
            means = norm_data['mean']
            stds = norm_data['std']
            col_from_inport = input_columns[self.INPUT_NORM_MODEL_NAME]
            cols = [i for i in col_from_inport.keys()]
            cols.sort()
        else:
            # need to compute the mean and std
            if self.conf.get('include', True):
                cols = self.conf['columns']
            else:
                cols = input_df.columns.difference(
                    self.conf['columns']).values.tolist()
            cols.sort()
            means = input_df[cols].mean()
            stds = input_df[cols].std()
        norm = (input_df[cols] - means) / stds
        col_dict = {i: norm[i] for i in cols}
        norm_df = input_df.assign(**col_dict)
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            output.update({self.OUTPUT_PORT_NAME: norm_df})
        if self.outport_connected(self.OUTPUT_NORM_MODEL_NAME):
            norm_data = {"mean": means, "std": stds}
            payload = NormalizationData(norm_data)
            output.update({self.OUTPUT_NORM_MODEL_NAME: payload})
        return output