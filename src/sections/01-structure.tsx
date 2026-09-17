import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import { MLP } from '../lib/network.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import { ArchitectureControls } from '../components/ui/ArchitectureControls.tsx';
import { Panel, Stats, Note } from '../components/ui/layout.tsx';
import { Detail, InWords, Notation } from '../components/ui/Detail.tsx';
import { Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';

export function StructureSection({ id, index }: SectionProps) {
  const [inputSize, setInputSize] = useState(3);
  const [outputSize, setOutputSize] = useState(2);
  const [hiddenUnits, setHiddenUnits] = useState<number[]>([5, 4]);

  const network = useMemo(
    () =>
      new MLP({
        inputSize,
        hiddenUnits,
        outputSize,
        hiddenActivation: 'relu',
        outputActivation: outputSize > 1 ? 'softmax' : 'sigmoid',
        loss: outputSize > 1 ? 'cce' : 'bce',
        seed: 7,
      }),
    [inputSize, hiddenUnits, outputSize],
  );

  const sizes = network.sizes;
  const rows = sizes.slice(1).map((units, l) => ({
    name: l === sizes.length - 2 ? 'Output' : `Hidden ${l + 1}`,
    shape: `${units} × ${sizes[l]}`,
    weights: units * sizes[l],
    biases: units,
  }));
  const totalParameters = network.parameterCount();

  const chain = useMemo(() => {
    const parts = [`\\mathbf{a}^{(0)} &= \\mathbf{x}`];
    for (let l = 1; l < sizes.length; l++) {
      const fn = l === sizes.length - 1 ? 'g' : 'f';
      parts.push(
        `\\mathbf{a}^{(${l})} &= ${fn}\\!\\left(W^{(${l})}\\mathbf{a}^{(${l - 1})} + \\mathbf{b}^{(${l})}\\right)`,
      );
    }
    parts.push(`\\hat{\\mathbf{y}} &= \\mathbf{a}^{(${sizes.length - 1})}`);
    return parts.join(' \\\\[4pt] ');
  }, [sizes]);

  return (
    <Section
      id={id}
      index={index}
      title="Network structure"
      lede={
        <>
          A feed-forward network is an ordered list of layers. Each layer holds a weight matrix and
          a bias vector, multiplies the previous layer's output, and applies a function elementwise.
          The architecture is fully described by the layer sizes and the choice of that function.
        </>
      }
    >
      <div className="grid grid--side">
        <Panel
          title="Architecture"
          hint="drag the sliders"
          caption="The diagram redraws from the layer sizes. Edge colour and thickness show the randomly initialised weights; no training has happened yet."
        >
          <NetworkDiagram
            network={network}
            height={300}
            showValues={false}
            inputLabels={Array.from({ length: inputSize }, (_, i) => `x${i + 1}`)}
            outputLabels={Array.from({ length: outputSize }, (_, i) => `ŷ${i + 1}`)}
          />
        </Panel>

        <div className="stack">
          <Panel title="Layer sizes">
            <div className="stack">
              <Slider
                label="Input features"
                min={1}
                max={5}
                step={1}
                value={inputSize}
                onChange={setInputSize}
                display={inputSize}
              />
              <ArchitectureControls hiddenUnits={hiddenUnits} onChange={setHiddenUnits} maxUnits={10} />
              <Slider
                label="Output units"
                min={1}
                max={4}
                step={1}
                value={outputSize}
                onChange={setOutputSize}
                display={outputSize}
              />
            </div>
          </Panel>

          <Stats
            items={[
              { label: 'Layers (weighted)', value: sizes.length - 1 },
              { label: 'Depth', value: hiddenUnits.length + 1 },
              { label: 'Parameters', value: totalParameters, accent: true },
            ]}
          />
        </div>
      </div>

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">The computation performed by one layer</h3>
          <p>
            Layer <M>{'l'}</M> holds a weight matrix <M>{'W^{(l)}'}</M> and a bias vector{' '}
            <M>{'\\mathbf{b}^{(l)}'}</M>. It receives the activations of the previous layer and
            produces its own:
          </p>
          <Equation caption="z is the pre-activation, a is the activation, f is applied elementwise.">
            {'\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}, \\qquad \\mathbf{a}^{(l)} = f\\!\\left(\\mathbf{z}^{(l)}\\right)'}
          </Equation>
          <InWords>
            <p>
              Take the list of numbers the previous layer produced. Multiply it by this layer's grid
              of weights, which gives one number per unit in this layer. Add each unit's bias. That
              list is <M>{'\\mathbf{z}'}</M>. Then run every entry of it through the function{' '}
              <M>{'f'}</M> separately, and the resulting list is <M>{'\\mathbf{a}'}</M> — what this
              layer hands on.
            </p>
            <p>
              The superscript <M>{'(l)'}</M> is a label saying which layer, not a power. Section 00
              has the full notation list.
            </p>
          </InWords>
          <p>
            If layer <M>{'l'}</M> has <M>{'n_l'}</M> units and the previous layer has{' '}
            <M>{'n_{l-1}'}</M>, then <M>{'W^{(l)}'}</M> has shape <M>{'n_l \\times n_{l-1}'}</M> and{' '}
            <M>{'\\mathbf{b}^{(l)}'}</M> has length <M>{'n_l'}</M>. The number of trainable
            parameters is the sum over layers:
          </p>
          <Equation>
            {'\\#\\text{params} = \\sum_{l=1}^{L} \\left( n_l\\, n_{l-1} + n_l \\right)'}
          </Equation>
          <InWords>
            <p>
              For each layer, count the weights — one for every pair of (unit in this layer, unit in
              the previous layer), which is width times previous width — then add one bias per unit.
              Do that for every layer and add the results together.
            </p>
          </InWords>
          <p>
            For the current architecture ({sizes.join(' → ')}) that is{' '}
            <strong className="mono">{totalParameters}</strong> numbers. Training means choosing
            values for all of them.
          </p>

          <h3 className="subhead">Why the layer has exactly this form</h3>
          <p>
            A layer is an affine map followed by a fixed non-linear function applied elementwise.
            Each of the three pieces does one job, and removing any of them removes a capability:
          </p>
          <ul>
            <li>
              <strong>The matrix <M>{'W'}</M></strong> mixes the previous layer's values. Row{' '}
              <M>{'j'}</M> is a direction in <M>{'\\mathbb{R}^{n_{l-1}}'}</M>, and{' '}
              <M>{'z_j'}</M> measures how far the incoming activation vector extends along it.
              Without <M>{'W'}</M> the units could not combine information from several inputs.
            </li>
            <li>
              <strong>The vector <M>{'\\mathbf{b}'}</M></strong> translates. An affine map{' '}
              <M>{'\\mathbf{x} \\mapsto W\\mathbf{x}'}</M> always sends{' '}
              <M>{'\\mathbf{0}'}</M> to <M>{'\\mathbf{0}'}</M>; adding{' '}
              <M>{'\\mathbf{b}'}</M> removes that constraint.
            </li>
            <li>
              <strong>The function <M>{'f'}</M></strong> is what stops the composition from
              collapsing. Without it, any number of layers is a single linear map.
            </li>
          </ul>

          <Detail title="A stack of linear layers is one linear layer">
            <p>
              Take two layers with no activation, so <M>{'f'}</M> is the identity:
            </p>
            <Equation plain>
              {'\\mathbf{a}^{(2)} = W^{(2)}\\left(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}\\right) + \\mathbf{b}^{(2)}'}
            </Equation>
            <p>Expanding the product and regrouping:</p>
            <Equation plain>
              {"\\mathbf{a}^{(2)} = \\underbrace{\\left(W^{(2)}W^{(1)}\\right)}_{W'}\\mathbf{x} + \\underbrace{\\left(W^{(2)}\\mathbf{b}^{(1)} + \\mathbf{b}^{(2)}\\right)}_{\\mathbf{b}'}"}
            </Equation>
            <p>
              <M>{"W'"}</M> is a single <M>{'n_2 \\times n_0'}</M> matrix and <M>{"\\mathbf{b}'"}</M> a
              single vector, so the two layers compute exactly what one layer of shape{' '}
              <M>{'n_2 \\times n_0'}</M> computes. By induction the same holds for any depth. A
              10-layer linear network has more parameters than a 1-layer one but represents no
              function the 1-layer one cannot, and in fact represents fewer: the product of two
              matrices of inner dimension <M>{'n_1'}</M> has rank at most{' '}
              <M>{'\\min(n_0, n_1, n_2)'}</M>, so a narrow hidden layer restricts the rank.
            </p>
          </Detail>

          <h3 className="subhead">Shapes, and what happens to a batch</h3>
          <p>
            The equations above describe one example. Frameworks process many at once. Stack{' '}
            <M>{'m'}</M> examples as the columns of{' '}
            <M>{'X \\in \\mathbb{R}^{n_0 \\times m}'}</M> and the same weights apply
            unchanged:
          </p>
          <Equation caption="The bias is added to every column — one vector, m copies.">
            {'Z^{(l)} = W^{(l)} A^{(l-1)} + \\mathbf{b}^{(l)}\\mathbf{1}^{\\top}, \\qquad A^{(l)} = f\\!\\left(Z^{(l)}\\right)'}
          </Equation>
          <p>
            Nothing about the parameters depends on <M>{'m'}</M>. That is why the same trained
            network runs on one example or ten thousand, and why the batch size is a choice about
            the optimiser rather than about the model.
          </p>

          <h3 className="subhead">Cost</h3>
          <p>
            Layer <M>{'l'}</M> performs <M>{'n_l n_{l-1}'}</M> multiply-accumulate operations per
            example, so a forward pass costs about <M>{'2\\sum_l n_l n_{l-1}'}</M> floating-point
            operations — roughly twice the parameter count. Backpropagation costs about twice that
            again, because it computes both the gradient with respect to the weights and the
            gradient flowing to the previous layer. A useful rule: one training step on one example
            costs roughly <M>{'6 \\times \\#\\text{params}'}</M> FLOPs. For the network above,{' '}
            <span className="mono">{totalParameters}</span> parameters gives about{' '}
            <span className="mono">{totalParameters * 6}</span> FLOPs per example per step.
          </p>
          <p>
            Memory splits the same way. Inference needs the parameters plus one layer of
            activations at a time. Training needs the parameters, the gradients (same size), and
            every activation from the forward pass held until the backward pass reaches it — which
            is why training a network takes several times the memory of running it.
          </p>
        </div>

        <div className="stack">
          <Panel title="Parameter count by layer" flush>
            <table className="data">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>W shape</th>
                  <th>Weights</th>
                  <th>Biases</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.name}>
                    <td>{row.name}</td>
                    <td>{row.shape}</td>
                    <td>{row.weights}</td>
                    <td>{row.biases}</td>
                  </tr>
                ))}
                <tr className="is-current">
                  <td>Total</td>
                  <td />
                  <td>{rows.reduce((sum, row) => sum + row.weights, 0)}</td>
                  <td>{rows.reduce((sum, row) => sum + row.biases, 0)}</td>
                </tr>
              </tbody>
            </table>
          </Panel>

          <Panel title="Full forward computation">
            <Equation plain>{`\\begin{aligned} ${chain} \\end{aligned}`}</Equation>
          </Panel>
        </div>
      </div>

      <div className="grid grid--side">
        <div className="prose-block">
          <h3 className="subhead">Notation used throughout</h3>
          <Notation
            entries={[
              { symbol: <M>{'L'}</M>, meaning: 'Number of weighted layers. The input is layer 0.' },
              { symbol: <M>{'n_l'}</M>, meaning: <>Number of units in layer <M>{'l'}</M>.</> },
              {
                symbol: <M>{'W^{(l)}'}</M>,
                meaning: (
                  <>
                    Weight matrix of layer <M>{'l'}</M>, shape <M>{'n_l \\times n_{l-1}'}</M>. The
                    entry <M>{'W^{(l)}_{ji}'}</M> connects unit <M>{'i'}</M> of layer{' '}
                    <M>{'l-1'}</M> to unit <M>{'j'}</M> of layer <M>{'l'}</M>.
                  </>
                ),
              },
              { symbol: <M>{'\\mathbf{b}^{(l)}'}</M>, meaning: <>Bias vector of layer <M>{'l'}</M>, length <M>{'n_l'}</M>.</> },
              { symbol: <M>{'\\mathbf{z}^{(l)}'}</M>, meaning: 'Pre-activations: the affine output, before f.' },
              { symbol: <M>{'\\mathbf{a}^{(l)}'}</M>, meaning: <>Activations: <M>{'f(\\mathbf{z}^{(l)})'}</M>. By convention <M>{'\\mathbf{a}^{(0)} = \\mathbf{x}'}</M>.</> },
              { symbol: <M>{'f, g'}</M>, meaning: 'Hidden activation and output activation. Different functions in general.' },
              { symbol: <M>{'\\hat{\\mathbf{y}}'}</M>, meaning: <>The network output, <M>{'\\mathbf{a}^{(L)}'}</M>.</> },
              { symbol: <M>{'\\mathbf{y}'}</M>, meaning: 'The target for the current example.' },
              { symbol: <M>{'L(\\hat{\\mathbf{y}}, \\mathbf{y})'}</M>, meaning: 'Loss on one example.' },
              { symbol: <M>{'J(\\theta)'}</M>, meaning: <>Objective: the loss averaged over the dataset, plus any regularisation term.</> },
              { symbol: <M>{'\\theta'}</M>, meaning: 'All parameters collectively — every weight and bias.' },
              { symbol: <M>{'\\delta^{(l)}'}</M>, meaning: <><M>{'\\partial L / \\partial \\mathbf{z}^{(l)}'}</M>, the quantity backpropagation carries.</> },
              { symbol: <M>{'\\eta'}</M>, meaning: 'Learning rate.' },
              { symbol: <M>{'\\odot'}</M>, meaning: 'Elementwise (Hadamard) product.' },
            ]}
          />
        </div>

        <div className="stack">
          <Note title="Terminology" accent>
            <p>
              Conventions differ on what counts as a layer. Here the input is layer 0 and is not
              counted as a weighted layer, because it holds no parameters. A network described as
              having <M>{'L'}</M> layers has <M>{'L'}</M> weight matrices. Its <em>depth</em> is the
              number of successive non-linear transformations applied between input and output —
              currently {hiddenUnits.length + 1}.
            </p>
          </Note>

          <Note title="Why 'fully connected'">
            <p>
              Every unit in layer <M>{'l'}</M> receives every unit of layer <M>{'l-1'}</M>, so{' '}
              <M>{'W^{(l)}'}</M> is dense. Other architectures impose structure on that matrix to
              encode an assumption: a convolutional layer ties weights together and zeroes distant
              entries, which encodes translation invariance and locality; an attention layer
              computes its mixing matrix from the data rather than storing it. Both are special
              cases of the same equation with constraints on <M>{'W'}</M>. Everything derived on
              this page — forward propagation, the chain rule, the update — applies unchanged to
              them.
            </p>
          </Note>

          <Note title="What the architecture fixes and what training chooses">
            <p>
              The layer sizes and the activations define a <em>family</em> of functions. Choosing{' '}
              <M>{'\\theta'}</M> picks one member of that family. Training never changes the
              family; it only searches inside it. If the target function is not in the family — a
              linear model asked to represent XOR — no amount of training will find it, and the
              failure shows as a training loss that stops falling while still high.
            </p>
          </Note>
        </div>
      </div>
    </Section>
  );
}
