from collections import OrderedDict
import importlib
import os
import networkx as nx
import yaml
import copy
from .node import Node

OUTPUT_ID = 'f291b900-bd19-11e9-aca3-a81e84f29b0f_uni_output'

DEFAULT_MODULE = os.getenv('GQUANT_PLUGIN_MODULE', "gquant.plugin_nodes")
MODLIB = importlib.import_module(DEFAULT_MODULE)

__all__ = ['TaskGraph', 'TaskSpecSchema', 'Task', 'OUTPUT_ID']


class TaskSpecSchema(object):
    '''Outline fields expected in a dictionary specifying a task node.

    :ivar id: unique id or name for the node
    :ivar type: Plugin class i.e. subclass of Node. Specified as string
        or subclass of Node
    :ivar conf: Configuration for the plugin i.e. parameterization. This is a
        dictionary.
    :ivar filepath: Path to python module for custom plugin types.
    :ivar inputs: List of ids of other tasks or an empty list.
    '''

    id = 'id'
    type = 'type'
    conf = 'conf'
    filepath = 'filepath'
    inputs = 'inputs'
    load = 'load'
    save = 'save'

    @classmethod
    def _typecheck(cls, schema_field, value):
        if (schema_field == cls.id):
            assert isinstance(value, str)
        elif schema_field == cls.type:
            assert (isinstance(value, str) or issubclass(value, Node))
        elif schema_field == cls.conf:
            assert (isinstance(value, dict) or isinstance(value, list))
        elif schema_field == cls.filepath:
            assert isinstance(value, str)
        elif schema_field == cls.inputs:
            assert isinstance(value, list)
            for item in value:
                assert isinstance(item, str)
        elif schema_field == cls.load:
            pass
        elif schema_field == cls.save:
            assert isinstance(value, bool)
        else:
            raise KeyError


class Task(object):
    ''' A strong typed Task class that is converted from dictionary.
    '''

    def __init__(self, task_spec):
        self._task_spec = {}  # internal dict
        self._task_spec[TaskSpecSchema.id] = task_spec[TaskSpecSchema.id]
        TaskSpecSchema._typecheck(TaskSpecSchema.id,
                                  self._task_spec[TaskSpecSchema.id])
        self._task_spec[TaskSpecSchema.type] = task_spec[TaskSpecSchema.type]
        TaskSpecSchema._typecheck(TaskSpecSchema.type, self._task_spec['type'])
        self._task_spec[TaskSpecSchema.conf] = task_spec[TaskSpecSchema.conf]
        TaskSpecSchema._typecheck(TaskSpecSchema.conf,
                                  self._task_spec[TaskSpecSchema.conf])
        if TaskSpecSchema.filepath in task_spec:
            self._task_spec[TaskSpecSchema.filepath] = task_spec[
                TaskSpecSchema.filepath]
            TaskSpecSchema._typecheck(TaskSpecSchema.filepath,
                                      self._task_spec[TaskSpecSchema.filepath])
        if TaskSpecSchema.load in task_spec:
            self._task_spec[TaskSpecSchema.load] =\
                task_spec[TaskSpecSchema.load]
            TaskSpecSchema._typecheck(TaskSpecSchema.load,
                                      self._task_spec[TaskSpecSchema.load])
        if TaskSpecSchema.save in task_spec:
            self._task_spec[TaskSpecSchema.save] =\
                task_spec[TaskSpecSchema.save]
            TaskSpecSchema._typecheck(TaskSpecSchema.save,
                                      self._task_spec[TaskSpecSchema.save])
        self._task_spec[TaskSpecSchema.inputs] =\
            task_spec[TaskSpecSchema.inputs]
        TaskSpecSchema._typecheck(TaskSpecSchema.inputs,
                                  self._task_spec[TaskSpecSchema.inputs])

    def __getitem__(self, key):
        return self._task_spec[key]

    def __setitem__(self, key, value):
        TaskSpecSchema._typecheck(key, value)
        self._task_spec[key] = copy.deepcopy(value)

    def get(self, key, default=None):
        try:
            return self._task_spec[key]
        except KeyError:
            return default

    def get_node_obj(self, replace=None):
        """
        instantiate a node instance for this task given the replacement setup

        Arguments
        -------
        replace: dict
            conf parameters replacement

        Returns
        -----
        object
            Node instance
        """
        task_spec = copy.deepcopy(self._task_spec)
        task_spec.update(replace)

        node_id = task_spec[TaskSpecSchema.id]
        modulepath = task_spec.get(TaskSpecSchema.filepath)

        node_type = task_spec[TaskSpecSchema.type]
        task = Task(task_spec)

        if isinstance(node_type, str):
            if modulepath is not None:
                spec = importlib.util.spec_from_file_location(node_id,
                                                              modulepath)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                NodeClass = getattr(mod, node_type)
            else:
                global MODLIB
                NodeClass = getattr(MODLIB, node_type)
        elif issubclass(node_type, Node):
            NodeClass = node_type
        else:
            raise "Not supported"

        node = NodeClass(task)
        return node


class TaskGraph(object):
    ''' TaskGraph class that is used to store the graph.
    '''

    __SETUP_YAML_ONCE = False

    @staticmethod
    def setup_yaml():
        '''Write out yaml in order for OrderedDict.'''
        # https://stackoverflow.com/a/8661021
        # https://stackoverflow.com/questions/47692094/lambda-works-
        # defined-function-does-not  # noqa
        # represent_dict_order = lambda dumper, data: \
        #     dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
        def represent_dict_order(dumper, data):
            return dumper.represent_mapping('tag:yaml.org,2002:map',
                                            data.items())
        yaml.add_representer(OrderedDict, represent_dict_order)

        TaskGraph.__SETUP_YAML_ONCE = True

    def __init__(self, objs=None):
        self.__task_list = []
        self.__index = 0
        if objs is not None:
            for obj in objs:
                self.__task_list.append(Task(obj))

    def __len__(self):
        return len(self.__task_list)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index == len(self.__task_list):
            raise StopIteration
        obj = self.__task_list[self.__index]
        self.__index += 1
        return obj

    def __find_roots(self, node, inputs, consider_load=True):
        """
        find the root nodes that the `node` dependes on

        Arguments
        -------
        node: Node
            the leaf node, of whom we need to find the dependent input nodes
        inputs: list
            resulting list to store all the root nodes in this list
        consider_load: bool
            whether it skips the node which are loading cache file or not
        Returns
        -----
        None

        """

        if (node.visited):
            return
        node.visited = True
        if len(node.inputs) == 0:
            inputs.append(node)
            return
        if consider_load and node.load:
            inputs.append(node)
            return
        for i in node.inputs:
            self.__find_roots(i, inputs, consider_load)

    @staticmethod
    def load_taskgraph(filename):
        """
        load the yaml file to TaskGraph object

        Arguments
        -------
        filename: str
            the filename pointing to the yaml file in the filesystem
        Returns
        -----
        object
            the TaskGraph instance

        """

        with open(filename) as f:
            obj = yaml.safe_load(f)
        t = TaskGraph(obj)
        return t

    def save_taskgraph(self, filename):
        """
        Write a list of tasks i.e. taskgraph to a yaml file.

        Arguments
        -------
        filename: str
            The filename to write a yaml file to.

        """

        if not TaskGraph.__SETUP_YAML_ONCE:
            TaskGraph.setup_yaml()

        # we want -id to be first in the resulting yaml file.
        tlist_od = []  # task list ordered
        for task in self.__task_list:
            tod = OrderedDict([(TaskSpecSchema.id, 'idholder'),
                               (TaskSpecSchema.type, 'typeholder'),
                               (TaskSpecSchema.conf, 'confholder'),
                               (TaskSpecSchema.inputs, 'inputsholder')
                               ])
            tod.update(task._task_spec)
            tlist_od.append(tod)

        with open(filename, 'w') as fh:
            yaml.dump(tlist_od, fh, default_flow_style=False)

    def viz_graph(self):
        """
        Generate the visulization of the graph in the JupyterLab

        Returns
        -----
        nx.DiGraph
        """
        G = nx.DiGraph()
        # instantiate objects
        for o in self.__task_list:
            for i in o[TaskSpecSchema.inputs]:
                G.add_edge(i, o[TaskSpecSchema.id])
        return G

    def build(self, replace=None):
        """
        compute the graph structure of the nodes. It will set the input and
        output nodes for each of the node

        Arguments
        -------
        replace: dict
            conf parameters replacement
        """
        self.__node_dict = {}
        replace = dict() if replace is None else replace
        # instantiate objects
        for task in self.__task_list:
            node = task.get_node_obj(replace.get(task['id'], {}))
            self.__node_dict[task['id']] = node

        # build the graph
        for task_id in self.__node_dict:
            node = self.__node_dict[task_id]
            for input_id in node._task_obj['inputs']:
                input_node = self.__node_dict[input_id]
                node.inputs.append(input_node)
                input_node.outputs.append(node)

        # this part is to do static type checks
        raw_inputs = []
        for k in self.__node_dict.keys():
            self.__find_roots(self.__node_dict[k], raw_inputs,
                              consider_load=False)

        for i in raw_inputs:
            i.columns_flow()

        # clean up the visited status for run computations
        for task_id in self.__node_dict:
            self.__node_dict[task_id].visited = False

    def __getitem__(self, key):
        return self.__node_dict[key]

    def __str__(self):
        out_str = ""
        for k in self.__node_dict.keys():
            out_str += k + ": " + str(self.__node_dict[k]) + "\n"
        return out_str

    def run(self, outputs, replace=None):
        """
        Flow the dataframes in the graph to do the data science computations.

        Arguments
        -------
        outputs: list
            a list of the leaf node IDs for which to return the final results
        replace: list
            a dict that defines the conf parameters replacement

        Returns
        -----
        tuple
            the results corresponding to the outputs list
        """
        replace = dict() if replace is None else replace
        self.build(replace)
        output_task = Task({TaskSpecSchema.id: OUTPUT_ID,
                            TaskSpecSchema.conf: {},
                            TaskSpecSchema.type: "dumpy",
                            TaskSpecSchema.inputs: []})
        output_node = Node(output_task)
        # want to save the intermediate results
        output_node.clear_input = False
        results = []
        results_obj = []
        for o in outputs:
            o_obj = self.__node_dict[o]
            results_obj.append(o_obj)
            output_node.inputs.append(o_obj)
            o_obj.outputs.append(output_node)

        inputs = []
        self.__find_roots(output_node, inputs, consider_load=True)
        # now clean up the graph, removed the node that is not used for
        # computation
        for key in self.__node_dict:
            current_obj = self.__node_dict[key]
            if not current_obj.visited:
                for i in current_obj.inputs:
                    i.outputs.remove(current_obj)
                current_obj.inputs = []

        for i in inputs:
            i.flow()

        for r_obj in results_obj:
            results.append(output_node.input_df[r_obj])

        # clean the results afterwards
        output_node.input_df = {}
        return tuple(results)


if __name__ == "__main__":
    t = {'id': 'test',
         'type': "DropNode",
         'conf': {},
         'inputs': ["node_other"]}
    task = Task(t)
