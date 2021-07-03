import warnings
import copy
import traceback
import cloudpickle
import base64
from types import ModuleType
from collections import OrderedDict
import ruamel.yaml

from .node import Node
from ._node_flow import OUTPUT_ID, OUTPUT_TYPE, _CLEANUP
from .task import Task
from .taskSpecSchema import TaskSpecSchema
from .portsSpecSchema import NodePorts, ConfSchema, PortsSpecSchema
from .util import get_encoded_class
from .config_nodes_modules import get_node_obj

__all__ = ['TaskGraph', 'OutputCollector']

server_task_graph = None


def add_module_from_base64(module_name, class_str):
    class_obj = cloudpickle.loads(base64.b64decode(class_str))
    class_name = class_obj.__name__
    import sys
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        mod = ModuleType(module_name)
        sys.modules[module_name] = mod
    setattr(mod, class_name, class_obj)
    return class_obj


class OutputCollector(Node):
    def meta_setup(self):
        return super().meta_setup()

    def ports_setup(self):
        return NodePorts(inports={}, outports={})

    def conf_schema(self):
        return ConfSchema()

    def process(self, inputs):
        return super().process(inputs)


class Results(object):

    def __init__(self, values):
        self.values = tuple([i[1] for i in values])
        self.__keys = tuple([i[0] for i in values])
        self.__dict = OrderedDict(values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.values[key]
        else:
            return self.__dict[key]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return "Results"+self.__dict.__repr__()[11:]

    def __str__(self):
        return "Results"+self.__dict.__str__()[11:]

    def __contains__(self, key):
        return True if key in self.__dict else False

    def get_keys(self):
        return self.__keys


def formated_result(result):
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import Output
    outputs = [Output() for i in range(len(result))]
    for i in range(len(result)):
        with outputs[i]:
            display(result[i])
    tab = widgets.Tab()
    tab.children = outputs
    for i in range(len(result)):
        tab.set_title(i, result.get_keys()[i])
    return tab


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
        ruamel.yaml.add_representer(OrderedDict, represent_dict_order)

        TaskGraph.__SETUP_YAML_ONCE = True

    def __init__(self, task_spec_list=None):
        '''
        :param task_spec_list: List of task-spec dicts per TaskSpecSchema.
        '''
        self.__task_list = {}
        self.__node_dict = {}
        self.__index = None
        # this is server widget that this taskgraph associated with
        self.__widget = None

        error_msg = 'Task-id "{}" already in the task graph. Set '\
                    'replace=True to replace existing task with extended task.'

        self.__extend(task_spec_list=task_spec_list, replace=False,
                      error_msg=error_msg)

    def __extend(self, task_spec_list=None, replace=False, error_msg=None):
        tspec_list = dict() if task_spec_list is None else task_spec_list

        if error_msg is None:
            error_msg = 'Task-id "{}" already in the task graph. Set '\
                        'replace=True to replace existing task.'

        for tspec in tspec_list:
            task = Task(tspec)
            task_id = task[TaskSpecSchema.task_id]
            if task_id in self.__task_list and not replace:
                raise Exception(error_msg.format(task_id))
            self.__task_list[task_id] = task

        if self.__widget is not None:
            self.__widget.value = self.export_task_speclist()

    def extend(self, task_spec_list=None, replace=False):
        '''
        Add more task-spec dicts to the graph

        :param task_spec_list: List of task-spec dicts per TaskSpecSchema.
        '''

        error_msg = 'Task-id "{}" already in the task graph. Set '\
                    'replace=True to replace existing task with extended task.'

        self.__extend(task_spec_list=task_spec_list, replace=replace,
                      error_msg=error_msg)

    def __contains__(self, task_id):
        return True if task_id in self.__task_list else False

    def __len__(self):
        return len(self.__task_list)

    def __iter__(self):
        self.__index = 0
        self.__tlist = list(self.__task_list.values())
        return self

    def __next__(self):
        idx = self.__index
        if idx is None or idx == len(self.__tlist):
            self.__index = None
            raise StopIteration
        task = self.__tlist[idx]
        self.__index = idx + 1
        return task

    def __find_roots(self, node, inputs, consider_load=True):
        """
        find the root nodes that the `node` depends on

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

        for node_in in node.inputs:
            inode = node_in['from_node']
            self.__find_roots(inode, inputs, consider_load)

    def start_labwidget(self):
        from IPython.display import display
        display(self.draw())

    @staticmethod
    def register_lab_node(module_name, class_obj):
        """
        Register the node class for the Greenflowlab. It put the class_obj
        into a sys.modules with `module_name`. It will register the node
        class into the Jupyterlab kernel space, communicate with the
         client to populate the add nodes menus, sync up with
         Jupyterlab Server space to register the node class.

        The latest registered `class_obj` overwrites the old one.

        Arguments
        -------
        module_name: str
            the module name for `class_obj`. It will also be the menu name for
             the node. Note, if use '.' inside the 'module_name', the client
              will automatically construct the hierachical menus based on '.'

        class_obj: Node
            The node class that is the subclass of greenflow 'Node'. It is
             usually defined dynamically so it can be registered.

        Returns
        -----
        None
        """
        global server_task_graph
        if server_task_graph is None:
            server_task_graph = TaskGraph()
            server_task_graph.start_labwidget()
        server_task_graph.register_node(module_name, class_obj)

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
            yaml = ruamel.yaml.YAML(typ='safe')
            yaml.constructor.yaml_constructors[
                u'tag:yaml.org,2002:timestamp'] = \
                yaml.constructor.yaml_constructors[u'tag:yaml.org,2002:str']
            obj = yaml.load(f)
        t = TaskGraph(obj)
        return t

    def export_task_speclist(self):
        tlist_od = []  # task list ordered
        for task in self:
            tod = OrderedDict([(TaskSpecSchema.task_id, 'idholder'),
                               (TaskSpecSchema.node_type, 'typeholder'),
                               (TaskSpecSchema.conf, 'confholder'),
                               (TaskSpecSchema.inputs, 'inputsholder')
                               ])
            tod.update(task._task_spec)
            if not isinstance(tod[TaskSpecSchema.node_type], str):
                tod[TaskSpecSchema.node_type] = tod[
                    TaskSpecSchema.node_type].__name__
            tlist_od.append(tod)
        return tlist_od

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
        tlist_od = self.export_task_speclist()
        with open(filename, 'w') as fh:
            ruamel.yaml.dump(tlist_od, fh, default_flow_style=False)

    def viz_graph(self, show_ports=False, pydot_options=None):
        """
        Generate the visulization of the graph in the JupyterLab

        Returns
        -----
        nx.DiGraph
        """
        import networkx as nx
        G = nx.DiGraph()
        if pydot_options:
            G.graph['graph'] = pydot_options
        # instantiate objects
        for itask in self:
            task_inputs = itask[TaskSpecSchema.inputs]
            to_task = itask[TaskSpecSchema.task_id]
            to_type = itask[TaskSpecSchema.node_type]
            if to_task == "":
                to_task = OUTPUT_TYPE
            for iport_or_tid in task_inputs:
                # iport_or_tid: it is either to_port or task id (tid) b/c
                #     if using ports API task_inputs is a dictionary otherwise
                #     task_inputs is a list.
                taskin_and_oport = task_inputs[iport_or_tid] \
                    if isinstance(task_inputs, dict) else iport_or_tid
                isplit = taskin_and_oport.split('.')
                from_task = isplit[0]
                from_port = isplit[1] if len(isplit) > 1 else None
                if show_ports and from_port is not None:
                    to_port = iport_or_tid
                    common_tip = taskin_and_oport
                    G.add_edge(from_task, common_tip, label=from_port)
                    G.add_edge(common_tip, to_task, label=to_port)
                    tnode = G.nodes[common_tip]
                    tnode.update({
                        # 'label': '',
                        'shape': 'point'})
                else:
                    G.add_edge(from_task, to_task)

            # draw output ports
            if show_ports:

                if (to_type == OUTPUT_TYPE):
                    continue
                task_node = get_node_obj(itask)
                # task_outputs = itask.get(TaskSpecSchema.outputs, [])
                for pout in task_node._get_output_ports():
                    out_tip = '{}.{}'.format(
                        itask[TaskSpecSchema.task_id], pout)
                    G.add_edge(to_task, out_tip, label=pout)
                    tnode = G.nodes[out_tip]
                    tnode.update({
                        # 'label': '',
                        'shape': 'point'})
        return G

    def _build(self, replace=None, profile=False):
        """
        compute the graph structure of the nodes. It will set the input and
        output nodes for each of the node

        Arguments
        -------
        replace: dict
            conf parameters replacement
        """
        self.__node_dict.clear()
        replace = dict() if replace is None else replace

        # check if there are item in the replace that is not in the graph
        task_ids = set([task[TaskSpecSchema.task_id] for task in self])
        for rkey in replace.keys():
            if rkey not in task_ids:
                warnings.warn(
                    'Replace task-id {} not found in task-graph'.format(rkey),
                    RuntimeWarning)

        # instantiate node objects
        for task in self:
            task_id = task[TaskSpecSchema.task_id]
            nodetype = task[TaskSpecSchema.node_type]
            if (task_id == OUTPUT_ID or nodetype == OUTPUT_TYPE):
                output_task = Task({
                    TaskSpecSchema.task_id: OUTPUT_ID,
                    TaskSpecSchema.conf: {},
                    TaskSpecSchema.node_type: OutputCollector,
                    TaskSpecSchema.inputs: task[TaskSpecSchema.inputs]
                })
                node = get_node_obj(output_task, tgraph_mixin=True)
            else:
                node = get_node_obj(task, replace.get(task_id), profile,
                                    tgraph_mixin=True)
            self.__node_dict[task_id] = node

        # build the graph
        for task_id in self.__node_dict:
            node = self.__node_dict[task_id]
            task_inputs = node._task_obj[TaskSpecSchema.inputs]
            for iport in task_inputs:
                # node_inputs should be a dict with entries:
                #     {iport: taskid.oport}
                input_task = task_inputs[iport].split('.')
                dst_port = iport

                input_id = input_task[0]
                # src_port = input_task[1] if len(input_task) > 1 else None
                src_port = input_task[1]

                try:
                    input_node = self.__node_dict[input_id]
                except KeyError:
                    raise LookupError(
                        'Missing task "{}". Add task spec to TaskGraph.'
                        .format(input_id))

                node.inputs.append({
                    'from_node': input_node,
                    'from_port': src_port,
                    'to_port': dst_port
                })
                # input_node.outputs.append(node)
                input_node.outputs.append({
                    'to_node': node,
                    'to_port': dst_port,
                    'from_port': src_port
                })

    def build(self, replace=None, profile=False):
        """
        compute the graph structure of the nodes. It will set the input and
        output nodes for each of the node

        Arguments
        -------
        replace: dict
            conf parameters replacement
        """
        # make connection only
        self._build(replace=replace, profile=profile)

        # Columns type checking is done in the :meth:`TaskGraph._run` after the
        # outputs are specified and participating tasks are determined.

        # this part is to update each of the node so dynamic inputs can be
        # processed
        self.breadth_first_update()

    def breadth_first_update(self, extra_roots=[], extra_updated=set()):
        """
        Do a breadth first graph traversal and update nodes.

        Update each note following the causal order. The children notes are
        only added to the queue if all the parents are updated.

        Each node is only updated once.

        extra_roots and extra_updated should be empty for normal graph. It
        is used for composite node when the graph is connected to other
        graph.
        """
        queue = []
        updated = extra_updated
        for k in self.__node_dict.keys():
            if len(self.__node_dict[k].inputs) == 0:
                queue.append(self.__node_dict[k])
        queue.extend(extra_roots)
        while (len(queue) != 0):
            node_to_update = queue.pop(0)
            # print('update {}'.format(node_to_update.uid))
            if node_to_update not in updated:
                node_to_update.update()
            updated.add(node_to_update)
            for element in node_to_update.outputs:
                child = element['to_node']
                if all([i['from_node'] in updated for i in child.inputs]):
                    queue.append(child)
        # print('----done----')

    def __getitem__(self, key):
        if not self.__node_dict:
            warnings.warn(
                'Task graph internal state empty. Did you build the task '
                'graph? Run ".build()"',
                RuntimeWarning)

        elif key not in self.__node_dict:
            warnings.warn(
                'Task graph missing task id "{}". Check the spelling of the '
                'task id.'.format(key),
                RuntimeWarning)

        return self.__node_dict[key]

    def __str__(self):
        out_str = ""
        for k in self.__node_dict.keys():
            out_str += k + ": " + str(self.__node_dict[k]) + "\n"
        return out_str

    def reset(self):
        self.__node_dict.clear()
        self.__task_list.clear()
        self.__index = None

    def register_node(self, module_name, classObj):
        """
        Check `TaskGraph.register_lab_node`
        """
        if self.__widget is not None:
            encoded_class = get_encoded_class(classObj)
            cacheCopy = copy.deepcopy(self.__widget.cache)
            cacheCopy['register'] = {
                "module": module_name,
                "class": encoded_class
            }
            add_module_from_base64(module_name, encoded_class)
            self.__widget.cache = cacheCopy

    def _run(self, outputs=None, replace=None, profile=False, formated=False):
        replace = dict() if replace is None else replace

        self.build(replace, profile)

        graph_outputs = []
        # add the output graph only if the Output Node is not in the graph
        found_output_node = False
        for task in self:
            if (task[TaskSpecSchema.task_id] == OUTPUT_ID or
                    task[TaskSpecSchema.node_type] == OUTPUT_TYPE):
                found_output_node = True
                outputs_collector_node = self[task[TaskSpecSchema.task_id]]
                for input_item in outputs_collector_node.inputs:
                    from_node_id = input_item['from_node'].uid
                    fromStr = from_node_id+'.'+input_item['from_port']
                    graph_outputs.append(fromStr)
                break

        if outputs is None:
            outputs = graph_outputs

        if not found_output_node:
            output_task = Task({
                TaskSpecSchema.task_id: OUTPUT_ID,
                TaskSpecSchema.conf: {},
                TaskSpecSchema.node_type: OutputCollector,
                TaskSpecSchema.inputs: []
            })

            outputs_collector_node = get_node_obj(output_task,
                                                  tgraph_mixin=True)

        outputs_collector_node.clear_input = False
        if not found_output_node or outputs is not None:
            if found_output_node:
                # clean all the connections to this output node
                for uid in self.__node_dict.keys():
                    node = self.__node_dict[uid]
                    node.outputs = list(filter(
                        lambda x: x['to_node'] != outputs_collector_node,
                        node.outputs))

            # remove the output
            # set the connection only if output_node is manullay created
            # or the output is overwritten
            outputs_collector_node.inputs.clear()
            outputs_collector_node.outputs.clear()
            for task_id in outputs:
                nodeid_oport = task_id.split('.')
                nodeid = nodeid_oport[0]
                oport = nodeid_oport[1] if len(nodeid_oport) > 1 else None
                onode = self.__node_dict[nodeid]
                dummy_port = task_id
                outputs_collector_node.inputs.append({
                    'from_node': onode,
                    'from_port': oport,
                    'to_port': dummy_port
                })
                onode.outputs.append({
                    'to_node': outputs_collector_node,
                    'to_port': dummy_port,
                    'from_port': oport
                })

        results_task_ids = outputs

        inputs = []
        self.__find_roots(outputs_collector_node, inputs, consider_load=True)

        # Validate metadata prior to running heavy compute
        for node in self.__node_dict.values():
            if not node.visited:
                continue

            # Run ports validation.
            PortsSpecSchema.validate_ports(node.ports_setup())
            node.validate_connected_ports()

            # Run meta setup in case the required meta are calculated
            # within the meta_setup and are NodeTaskGraphMixin dependent.
            # node.meta_setup()
            node.validate_connected_metadata()

        if self.__widget is not None:
            def progress_fun(uid):
                cacheCopy = copy.deepcopy(self.__widget.cache)
                nodes = list(filter(lambda x: x['id'] == uid,
                                    cacheCopy['nodes']
                                    if 'nodes' in cacheCopy else []))
                if len(nodes) > 0:
                    current_node = nodes[0]
                    current_node['busy'] = True
                self.__widget.cache = cacheCopy
            for i in inputs:
                i.flow(progress_fun)
            # clean up the progress

            def cleanup():
                import time
                cacheCopy = copy.deepcopy(self.__widget.cache)
                for node in cacheCopy.get('nodes', []):
                    node['busy'] = False
                time.sleep(1)
                self.__widget.cache = cacheCopy
            import threading
            t = threading.Thread(target=cleanup)
            t.start()
        else:
            for i in inputs:
                i.flow()

        results_dfs_dict = outputs_collector_node.input_df
        port_map = {}
        for input_item in outputs_collector_node.inputs:
            from_node_id = input_item['from_node'].uid
            fromStr = from_node_id+'.'+input_item['from_port']
            port_map[fromStr] = input_item['to_port']
        results = []
        for task_id in results_task_ids:
            results.append((task_id, results_dfs_dict[port_map[task_id]]))
        # clean the results afterwards
        outputs_collector_node.input_df = {}
        result = Results(results)
        ####
        # this is for nemo work around, to clean up the nemo graph
        self.run_cleanup()

        if formated:
            return formated_result(result)
        else:
            return result

    def run_cleanup(self, ui_clean=False):
        for v in _CLEANUP.values():
            v(ui_clean)

    def run(self, outputs=None, replace=None, profile=False, formated=False):
        """
        Flow the dataframes in the graph to do the data science computations.

        Arguments
        -------
        outputs: list
            a list of the leaf node IDs for which to return the final results
        replace: list
            a dict that defines the conf parameters replacement
        profile: Boolean
            whether profile the processing time of the nodes or not

        Returns
        -----
        tuple
            the results corresponding to the outputs list
        """
        if formated:
            # cap_run = out.capture(clear_output=True)(self._run)
            # result = cap_run(outputs=outputs, replace=replace,
            #                 profile=profile, formated=formated)
            try:
                err = ""
                result = None
                result = self._run(outputs=outputs, replace=replace,
                                   profile=profile, formated=formated)
            except Exception:
                err = traceback.format_exc()
            finally:
                import ipywidgets
                out = ipywidgets.Output(layout={'border': '1px solid black'})
                out.append_stderr(err)
            if result is None:
                result = ipywidgets.Tab()
            result.set_title(len(result.children), 'std output')
            result.children = result.children + (out,)
            return result
        else:
            return self._run(outputs=outputs, replace=replace, profile=profile,
                             formated=formated)

    def to_pydot(self, show_ports=False, pydot_options=None):
        import networkx as nx
        nx_graph = self.viz_graph(show_ports=show_ports,
                                  pydot_options=pydot_options)
        to_pydot = nx.drawing.nx_pydot.to_pydot
        pdot = to_pydot(nx_graph)
        return pdot

    def get_widget(self):
        if self.__widget is None:
            from greenflowlab.greenflowmodel import GreenflowWidget
            widget = GreenflowWidget()
            widget.value = self.export_task_speclist()
            widget.set_taskgraph(self)
            self.__widget = widget
        return self.__widget

    def draw(self, show='lab', fmt='png', show_ports=False,
             pydot_options=None):
        '''
        :param show: str; One of 'ipynb', 'lab'
        :param fmt: str; 'png' or 'svg'. Only used if show='ipynb'
        :param show_ports: boolean; Labels intermediate ports between nodes in
            the taskgraph. Only used if show='ipynb'
        :param pydot_options: dict; Passed to the graph attribute of a graphviz
            generated dot graph. Only used when show='ipynb'. Refer to:
                https://graphviz.org/doc/info/attrs.html
            Example: pydot_options={'rankdir': 'LR'} to draw left-to-right
        '''
        if show in ('ipynb',):
            pdot = self.to_pydot(show_ports, pydot_options)
            pdot_out = pdot.create(format=fmt)
            if fmt in ('svg',):
                from IPython.display import SVG as Image  # @UnusedImport
            else:
                from IPython.display import Image  # @Reimport
            plt = Image(pdot_out)
            return plt
        else:
            widget = self.get_widget()
            return widget
