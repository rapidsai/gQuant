import {
  DocumentWidget,
  DocumentRegistry,
  ABCWidgetFactory
} from '@jupyterlab/docregistry';
import { MainView } from './mainComponent';
import { requestAPI } from './gquantlab';
import YAML from 'yaml';
import { Signal, ISignal } from '@lumino/signaling';
import { MainAreaWidget } from '@jupyterlab/apputils';

export class ContentHandler {
  context: DocumentRegistry.Context;
  private _contentChanged = new Signal<ContentHandler, IChartInput>(this);

  get contentChanged(): ISignal<ContentHandler, IChartInput> {
    return this._contentChanged;
  }

  constructor(context: DocumentRegistry.Context) {
    this.context = context;
    //this.context.model.contentChanged.connect(this._onContentChanged, this);
    this.emit();
  }

  public update(content: string): void {
    this.context.model.fromString(content);
  }

  public emit(): void {
    this._onContentChanged();
  }

  private _onContentChanged(): void {
    console.log('content chagned');
    const refreshContent = async (): Promise<void> => {
      await this.context.ready;
      const yamlContent = this.context.model.toString();
      console.log('model path', this.context.path);
      const objContent = YAML.parse(yamlContent);
      const jsonString = JSON.stringify(objContent);
      // this.context.model.contentChanged.connect(this._onContentChanged, this);
      const allNodes = await requestAPI<any>('all_nodes');
      const workflows = await requestAPI<any>('load_graph', {
        body: jsonString,
        method: 'POST'
      });
      this._contentChanged.emit({
        allNodes: allNodes,
        nodes: workflows['nodes'],
        edges: workflows['edges']
      });
    };
    refreshContent();
  }
}

export interface IPort {
  name: string;
  type: string[];
}

export interface INode {
  parentIds?: string[];
  width?: number;
  x?: number;
  y?: number;
  id: string;
  type: string;
  schema: any;
  ui: any;
  conf: any;
  inputs: IPort[];
  outputs: IPort[];
  filepath?: string;
  required: any;
  output_columns: any;
}

export interface IEdge {
  from: string;
  to: string;
}

export interface IChartInput {
  allNodes: IAllNodes;
  nodes: INode[];
  edges: IEdge[];
}

export interface IAllNodes {
  [key: string]: INode[];
}

export class GquantWidget extends DocumentWidget<MainAreaWidget> {
  constructor(options: DocumentWidget.IOptions<MainAreaWidget>) {
    super({ ...options });
    this.context = options['context'];
  }

  readonly context: DocumentRegistry.Context;
}

/**
 * A widget factory for drawio.
 */
export class GquantFactory extends ABCWidgetFactory<
  GquantWidget,
  DocumentRegistry.IModel
> {
  /**
   * Create a new widget given a context.
   */
  constructor(options: DocumentRegistry.IWidgetFactoryOptions) {
    super(options);
  }

  protected createNewWidget(context: DocumentRegistry.Context): GquantWidget {
    const contentHandler = new ContentHandler(context);
    const mainView = new MainView(contentHandler);
    const widget = new MainAreaWidget<MainView>({ content: mainView });
    return new GquantWidget({ context, content: widget });
  }
}