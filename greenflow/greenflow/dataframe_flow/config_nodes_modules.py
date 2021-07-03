import os
import sys
from pathlib import Path
import copy
import pkgutil
import inspect
from collections import namedtuple
import configparser
import importlib

from .util import get_file_path
from .taskSpecSchema import TaskSpecSchema
from .task import Task
from ._node import _Node
from ._node_flow import NodeTaskGraphMixin


DEFAULT_MODULE = os.getenv('GREENFLOW_PLUGIN_MODULE', "greenflow.plugin_nodes")

MODULE_CACHE = {}


class ConfigParser(configparser.ConfigParser):
    """Can get options() without defaults
    """

    def options(self, section, no_defaults=True, **kwargs):
        if no_defaults:
            try:
                return list(self._sections[section].keys())
            except KeyError:
                raise configparser.NoSectionError(section)
        else:
            return super().options(section, **kwargs)


def get_greenflow_config_modules():
    if 'GREENFLOW_CONFIG' not in os.environ:
        os.environ['GREENFLOW_CONFIG'] = os.getcwd() + '/greenflowrc'
        print('\nGREENFLOW_CONFIG NOT SET. SETTING TO: {}'
              .format(os.environ['GREENFLOW_CONFIG']))

    config = ConfigParser(defaults=os.environ)
    greenflow_cfg = os.getenv('GREENFLOW_CONFIG', None)
    if Path(greenflow_cfg).is_file():
        config.read(greenflow_cfg)

    if 'ModuleFiles' not in config:
        return {}

    modules_names = config.options('ModuleFiles')
    modules_list = {imod: config['ModuleFiles'][imod]
                    for imod in modules_names}

    return modules_list


# create a task to add path path
def append_path(path):
    if path not in sys.path:
        sys.path.append(path)


def import_submodules(package, recursive=True, _main_package=None):
    """Import all submodules of a module, recursively, including subpackages.
    Finds members of those packages. If a member is a greenflow Node subclass
     then sets the top level package attribute with the class. This is done so
      that the class can be accessed via:
        NodeClass = getattr(mod, node_type)
    Where mod is the package.

    :param package: package (name or actual module)
    :type package: module, str
    """
    if isinstance(package, str):
        package = importlib.import_module(package)

    _main_package = package if _main_package is None else _main_package
    # for loader, name, is_pkg
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        mod = importlib.import_module(full_name)
        if recursive and is_pkg:
            # import_submodules(full_name, _main_package=_main_package)
            import_submodules(mod, _main_package=_main_package)

        for node in inspect.getmembers(mod):
            nodecls = node[1]
            if not inspect.isclass(nodecls):
                continue
            if not issubclass(nodecls, _Node):
                continue
            if nodecls.__name__ == 'Node':
                continue

            try:
                getattr(_main_package, nodecls.__name__)
            except AttributeError:
                setattr(_main_package, nodecls.__name__, nodecls)


def load_modules(pathfile, name=None):
    """
    Given a py filename with path information,
    It will load the file as a python
    module, put it into the sys.modules and add the path into the pythonpath.
    @param modulefile
        string, file name
    @returns
        namedtuple, absolute path and loaded module
    """
    key = (pathfile, name)
    if key in MODULE_CACHE:
        return MODULE_CACHE[key]
    filename = Path(pathfile)
    if not filename.exists():
        filename = get_file_path(str(filename))
        filename = Path(filename)

    if name is None:
        modulename = filename.stem
    else:
        modulename = name

    module_dir = str(filename.parent.absolute())

    if filename.is_dir():
        modulepath = filename.joinpath('__init__.py')
        modulename = filename.stem
    else:
        modulepath = filename

    spec = importlib.util.spec_from_file_location(modulename, str(modulepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    if filename.is_dir():
        import_submodules(mod)

    spec.loader.exec_module(mod)
    Load = namedtuple("Load", "path mod")
    loaded = Load(module_dir, mod)
    MODULE_CACHE[key] = loaded
    return loaded


def get_node_tgraphmixin_instance(NodeClass, task):
    '''Instantiate a node using task and NodeClass with NodeTaskGraphMixin
    mixin. The returned node instance is a task graph aware node meaning it
    can query task graph state such as its connections. Refer to the API of
    NodeTaskGraphMixin.

    :type NodeClass: a class type of a Node implementation
    :type task: an instance of class::Task

    '''

    class NodeInTaskGraph(NodeTaskGraphMixin, NodeClass):

        def __init__(self, task):
            NodeClass.__init__(self, task)
            NodeTaskGraphMixin.__init__(self)

        def __repr__(self):
            '''Override repr to show the name and path of the plugin
            node class.'''
            return '<{} {}.{} object at {}>'.format(
                self.__class__.__name__,
                NodeClass.__module__,
                NodeClass.__name__,
                hex(id(self)))

    return NodeInTaskGraph(task)


def get_node_obj(task, replace=None, profile=False, tgraph_mixin=False,
                 dask_ray_setup=True):
    """
    Instantiate a node instance for a task given the replacement setup.

    Arguments
    -------
    replace: dict
        conf parameters replacement
    profile: Boolean
        profile the node computation

    Returns
    -----
    object
        Node instance
    """
    replace = dict() if replace is None else replace

    task_spec = copy.copy(task._task_spec)
    task_spec.update(replace)

    # node_id = task_spec[TaskSpecSchema.task_id]
    modulepath = task_spec.get(TaskSpecSchema.filepath)
    module_name = task_spec.get(TaskSpecSchema.module)

    node_type = task_spec[TaskSpecSchema.node_type]
    task = Task(task_spec)

    NodeClass = None
    module_dir = None
    if isinstance(node_type, str):
        modules = get_greenflow_config_modules()
        if modulepath is not None:
            loaded = load_modules(modulepath)
            module_dir = loaded.path
            mod = loaded.mod
            NodeClass = getattr(mod, node_type)
        elif (module_name is not None):
            if module_name in sys.modules:
                mod = sys.modules[module_name]
            else:
                loaded = load_modules(
                    modules[module_name], name=module_name)
                module_dir = loaded.path
                mod = loaded.mod

            try:
                NodeClass = getattr(mod, node_type)
            except AttributeError:
                pass
        else:
            try:
                global DEFAULT_MODULE
                plugmod = os.getenv('GREENFLOW_PLUGIN_MODULE',
                                    DEFAULT_MODULE)
                MODLIB = importlib.import_module(plugmod)
                NodeClass = getattr(MODLIB, node_type)
            except AttributeError:
                for key in modules:
                    loaded = load_modules(modules[key], name=key)
                    module_dir = loaded.path
                    mod = loaded.mod
                    try:
                        NodeClass = getattr(mod, node_type)
                        break
                    except AttributeError:
                        continue

        if NodeClass is None:
            raise Exception("Cannot find the Node Class:" +
                            node_type)

        if module_dir:
            append_path(module_dir)

        if module_dir and dask_ray_setup:
            try:
                # Add python path to all the client workers
                # assume all the workers share the same directory
                # structure
                import dask.distributed
                client = dask.distributed.client.default_client()
                client.run(append_path, module_dir)
            except (ValueError, ImportError):
                pass

            try:
                import ray

                def ray_append_path(worker):
                    import sys  # @Reimport
                    if module_dir not in sys.path:
                        sys.path.append(module_dir)

                # TODO: This could be a Ray Driver functionality. Add
                #     module path to all workers.
                ray.worker.global_worker.run_function_on_all_workers(
                    ray_append_path)
            except (ValueError, ImportError):
                pass

    elif issubclass(node_type, _Node):
        NodeClass = node_type
    else:
        raise Exception("Node type not supported: {}".format(node_type))

    assert issubclass(NodeClass, _Node), \
        'Node-type is not a subclass of "Node" class.'

    if tgraph_mixin:
        node = get_node_tgraphmixin_instance(NodeClass, task)
    else:
        node = NodeClass(task)

    node.profile = profile

    return node
