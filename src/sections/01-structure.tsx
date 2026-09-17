import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import { MLP } from '../lib/network.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import { ArchitectureControls } from '../components/ui/ArchitectureControls.tsx';
import { Panel, Stats, Note } from '../components/ui/layout.tsx';
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
          <p>
            If layer <M>{'l'}</M> has <M>{'n_l'}</M> units and the previous layer has{' '}
            <M>{'n_{l-1}'}</M>, then <M>{'W^{(l)}'}</M> has shape <M>{'n_l \\times n_{l-1}'}</M> and{' '}
            <M>{'\\mathbf{b}^{(l)}'}</M> has length <M>{'n_l'}</M>. The number of trainable
            parameters is the sum over layers:
          </p>
          <Equation>
            {'\\#\\text{params} = \\sum_{l=1}^{L} \\left( n_l\\, n_{l-1} + n_l \\right)'}
          </Equation>
          <p>
            For the current architecture ({sizes.join(' → ')}) that is{' '}
            <strong className="mono">{totalParameters}</strong> numbers. Training means choosing
            values for all of them.
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

      <Note title="Terminology" accent>
        <p>
          Conventions differ on what counts as a layer. Here the input is layer 0 and is not counted
          as a weighted layer, because it holds no parameters. A network described as having{' '}
          <M>{'L'}</M> layers has <M>{'L'}</M> weight matrices. Its <em>depth</em> is the number of
          successive non-linear transformations applied between input and output — currently{' '}
          {hiddenUnits.length + 1}.
        </p>
      </Note>
    </Section>
  );
}
