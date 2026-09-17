import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { DATASETS, DATASET_NAMES, generateDataset } from '../lib/datasets.ts';
import type { DatasetName } from '../lib/datasets.ts';
import { ACTIVATIONS, HIDDEN_ACTIVATIONS } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { useEnsembleTrainer } from '../hooks/useEnsembleTrainer.ts';
import type { EnsembleMember } from '../hooks/useEnsembleTrainer.ts';
import { DecisionBoundary } from '../components/viz/DecisionBoundary.tsx';
import { MetricChart } from '../components/viz/MetricChart.tsx';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Detail, InWords } from '../components/ui/Detail.tsx';
import { Button, SelectField, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { fmt, fmtPercent } from '../lib/format.ts';

const SERIES_COLORS = ['#94a3b8', '#e0761f', '#3b82f6', '#16a34a'];

/** Exact upper bound on the linear regions of a one-hidden-layer ReLU net in d inputs. */
function shallowRegions(units: number, inputs = 2): number {
  let total = 0;
  for (let j = 0; j <= inputs; j++) {
    let binomial = 1;
    for (let k = 0; k < j; k++) binomial = (binomial * (units - k)) / (k + 1);
    total += binomial;
  }
  return Math.round(total);
}

/** Lower bound for an L-hidden-layer ReLU net of width n on d inputs. */
function deepRegions(units: number, layers: number, inputs = 2): number {
  const base = Math.floor(units / inputs) ** inputs;
  return Math.round(base ** (layers - 1) * shallowRegions(units, inputs));
}

export function DepthSection({ id, index }: SectionProps) {
  const [datasetName, setDatasetName] = useState<DatasetName>('spiral');
  const [width, setWidth] = useState(6);
  const [activation, setActivation] = useState<ActivationName>('relu');
  const [learningRate, setLearningRate] = useState(0.25);
  const [selected, setSelected] = useState(2);
  const [counterWidth, setCounterWidth] = useState(8);

  const train = useMemo(
    () => generateDataset(datasetName, { count: 240, noise: 0.05, seed: 23 }),
    [datasetName],
  );

  const members = useMemo<EnsembleMember[]>(() => {
    const base: Omit<NetworkConfig, 'hiddenUnits'> = {
      inputSize: 2,
      outputSize: 1,
      hiddenActivation: activation,
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 808,
    };
    return [
      { label: 'Input → Output', config: { ...base, hiddenUnits: [] } },
      { label: 'Input → H → Output', config: { ...base, hiddenUnits: [width] } },
      { label: 'Input → H → H → Output', config: { ...base, hiddenUnits: [width, width] } },
      { label: 'Input → H → H → H → Output', config: { ...base, hiddenUnits: [width, width, width] } },
    ];
  }, [activation, width]);

  const trainer = useEnsembleTrainer(members, train, learningRate, 16, 2, 1500);

  const lossSeries = useMemo(
    () =>
      trainer.results.map((result, i) => ({
        label: result.label,
        color: SERIES_COLORS[i],
        values: result.history,
      })),
    [trainer.results],
  );

  const selectedResult = trainer.results[selected];
  const hiddenLayerIndex = selectedResult.network.sizes.length - 2;
  const hiddenCount = hiddenLayerIndex >= 1 ? selectedResult.network.sizes[hiddenLayerIndex] : 0;

  return (
    <Section
      id={id}
      index={index}
      title="Why depth matters"
      lede={
        <>
          Four networks, identical except for the number of hidden layers, trained on the same data
          with the same learning rate and the same number of epochs. The difference in what they can
          represent is visible in the boundaries they produce.
        </>
      }
    >
      <div className="grid grid--side">
        <Panel
          title="Same data, increasing depth"
          hint={`epoch ${trainer.epoch}`}
          caption="All four start from the same random seed and receive the same updates per epoch. Only the number of hidden layers differs."
        >
          <div className="grid grid--4" style={{ gap: 12 }}>
            {trainer.results.map((result, i) => (
              <button
                key={result.label}
                type="button"
                onClick={() => setSelected(i)}
                style={{
                  border: i === selected ? '1px solid var(--accent)' : '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  background: 'transparent',
                  padding: 6,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <DecisionBoundary
                  predict={(x1, x2) => result.network.forward([x1, x2]).output[0]}
                  train={train}
                  version={trainer.version}
                  resolution={48}
                />
                <div style={{ marginTop: 6 }}>
                  <div className="mono" style={{ fontSize: 11, color: 'var(--text)' }}>
                    {result.network.sizes.join(' → ')}
                  </div>
                  <div className="mono" style={{ fontSize: 11, color: 'var(--text-faint)' }}>
                    loss {fmt(result.loss, 3)} · {fmtPercent(result.accuracy, 0)} ·{' '}
                    {result.parameterCount}p
                  </div>
                </div>
              </button>
            ))}
          </div>
        </Panel>

        <div className="stack">
          <Panel title="Run">
            <div className="btn-row">
              <Button variant="primary" onClick={() => (trainer.running ? trainer.pause() : trainer.start())}>
                {trainer.running ? 'Pause' : 'Train all four'}
              </Button>
              <Button onClick={() => trainer.runEpochs(25)}>+25 epochs</Button>
              <Button onClick={trainer.reset}>Reset</Button>
            </div>
            <div className="stack stack--sm" style={{ marginTop: 14 }}>
              <SelectField
                label="Dataset"
                value={datasetName}
                onChange={setDatasetName}
                options={DATASET_NAMES.map((name) => ({ value: name, label: DATASETS[name].label }))}
              />
              <Slider label="Units per hidden layer" min={2} max={12} step={1} value={width} onChange={setWidth} display={width} />
              <Slider
                label="Learning rate η"
                min={0.01}
                max={1}
                step={0.01}
                value={learningRate}
                onChange={setLearningRate}
                display={fmt(learningRate, 2)}
              />
              <Segmented
                label="Activation"
                value={activation}
                options={HIDDEN_ACTIVATIONS.map((n) => ({ value: n, label: ACTIVATIONS[n].label }))}
                onChange={setActivation}
              />
            </div>
          </Panel>

          <Panel title="Training loss, all four" flush>
            <div style={{ padding: '10px 10px 0' }}>
              <MetricChart series={lossSeries} yMin={0} yLabel="loss" height={200} ariaLabel="Loss by depth" />
            </div>
            <div className="figure__caption">
              {trainer.results.map((result, i) => (
                <span key={result.label} style={{ marginRight: 14, whiteSpace: 'nowrap' }}>
                  <span
                    className="legend__swatch"
                    style={{ background: SERIES_COLORS[i], display: 'inline-block', marginRight: 5 }}
                  />
                  {result.network.sizes.length - 1} layer{result.network.sizes.length - 1 === 1 ? '' : 's'}
                </span>
              ))}
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid grid--side">
        <Panel
          title={`Hidden unit responses — ${selectedResult.label}`}
          hint="last hidden layer"
          caption="Each tile shows one unit's activation across the input plane; the dark curve marks where that unit's output is zero. These are the features the next layer combines. Click a boundary above to switch network."
        >
          {hiddenCount === 0 ? (
            <p className="faint" style={{ fontSize: 13, margin: 0 }}>
              This network has no hidden layer. Its output is a single sigmoid applied to a linear
              function of the inputs, so there are no intermediate features to show.
            </p>
          ) : (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${Math.min(6, hiddenCount)}, minmax(0, 1fr))`,
                gap: 8,
              }}
            >
              {Array.from({ length: hiddenCount }, (_, unit) => (
                <div key={unit}>
                  <DecisionBoundary
                    predict={(x1, x2) => {
                      const trace = selectedResult.network.forward([x1, x2]);
                      const a = trace.layers[hiddenLayerIndex - 1].a[unit];
                      return 1 / (1 + Math.exp(-2 * a));
                    }}
                    train={[]}
                    version={trainer.version}
                    resolution={36}
                    showPoints={false}
                  />
                  <div className="mono faint" style={{ fontSize: 10, textAlign: 'center', marginTop: 2 }}>
                    a{unit + 1}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <div className="prose-block">
          <h3 className="subhead">Depth as composition</h3>
          <p>
            A network is a composition <M>{'f_L \\circ \\cdots \\circ f_1'}</M>. Each{' '}
            <M>{'f_l'}</M> is an affine map followed by a non-linearity. With ReLU, every{' '}
            <M>{'f_l'}</M> is piecewise linear, so the whole network is piecewise linear: the input
            space is partitioned into regions, and on each region the network is exactly an affine
            function.
          </p>
          <p>
            A single hidden layer with <M>{'n'}</M> units places <M>{'n'}</M> hyperplanes in the
            input space. In <M>{'d = 2'}</M> dimensions they cut the plane into at most
          </p>
          <Equation plain>{'\\binom{n}{0} + \\binom{n}{1} + \\binom{n}{2}'}</Equation>
          <InWords tag="notation">
            <p>
              <M>{'\\binom{n}{k}'}</M>, read "n choose k", counts how many ways you can pick{' '}
              <M>{'k'}</M> items from <M>{'n'}</M> when the order does not matter.{' '}
              <M>{'\\binom{n}{0} = 1'}</M> (one way to pick nothing),{' '}
              <M>{'\\binom{n}{1} = n'}</M>, and <M>{'\\binom{n}{2} = n(n-1)/2'}</M>.
            </p>
            <p>
              For 8 units: <M>{'1 + 8 + 28 = 37'}</M> regions. Doubling the width to 16 gives{' '}
              <M>{'1 + 16 + 120 = 137'}</M> — roughly four times as many for twice the units, which
              is what "quadratic in the width" means.
            </p>
          </InWords>
          <p>
            regions — quadratic in <M>{'n'}</M>. Adding a second layer does not add hyperplanes; it
            folds the regions produced by the first layer onto each other, so each subsequent layer
            multiplies the region count rather than adding to it. The growth becomes exponential in
            depth and polynomial in width.
          </p>

          <Detail kicker="derivation" title="Counting the regions of one hidden layer">
            <p>
              Each unit's boundary is a hyperplane in <M>{'\\mathbb{R}^{d}'}</M>. The question is
              how many pieces <M>{'n'}</M> hyperplanes in general position cut the space into. Let{' '}
              <M>{'r(n, d)'}</M> be that number.
            </p>
            <p>
              Add the <M>{'n'}</M>-th hyperplane to an arrangement of{' '}
              <M>{'n-1'}</M>. It intersects each existing region it passes through and splits it in
              two, so the increase equals the number of regions it meets. Those regions correspond
              exactly to the regions that the other <M>{'n-1'}</M> hyperplanes cut the new
              hyperplane into — and the new hyperplane is itself a space of dimension{' '}
              <M>{'d-1'}</M>. So
            </p>
            <Equation plain>{'r(n, d) = r(n-1, d) + r(n-1, d-1)'}</Equation>
            <p>
              With the base cases <M>{'r(0,d) = 1'}</M> and{' '}
              <M>{'r(n,0) = 1'}</M>, this recursion has the closed-form solution
            </p>
            <Equation plain>{'r(n, d) = \\sum_{j=0}^{d} \\binom{n}{j}'}</Equation>
            <p>
              which can be checked by Pascal's rule{' '}
              <M>{'\\binom{n}{j} = \\binom{n-1}{j} + \\binom{n-1}{j-1}'}</M>. For{' '}
              <M>{'d = 2'}</M> this is{' '}
              <M>{'1 + n + \\binom{n}{2}'}</M>: two lines give 4 regions, three give 7, eight give
              37 — the number in the first row of the table below.
            </p>
            <p>
              The count is <M>{'\\Theta(n^{d})'}</M> for fixed <M>{'d'}</M> — polynomial in the
              width. Since a one-hidden-layer network of width{' '}
              <M>{'n'}</M> has about <M>{'n(d+2)'}</M> parameters, the regions grow only
              polynomially in the parameter count.
            </p>
          </Detail>

          <Detail title="Why a second layer multiplies rather than adds">
            <p>
              Consider a single ReLU unit in one dimension with{' '}
              <M>{'a = |z|'}</M> — achievable as the sum of two ReLUs,{' '}
              <M>{'\\max(0,z) + \\max(0,-z)'}</M>. This map is two-to-one: the points{' '}
              <M>{'z'}</M> and <M>{'-z'}</M> are sent to the same output. Anything the next layer
              computes downstream is therefore applied identically to both, so whatever pattern the
              next layer draws in its input space appears <em>twice</em> in the original input
              space, mirrored.
            </p>
            <p>
              A layer of <M>{'n'}</M> such units in <M>{'d'}</M> dimensions can fold the space along{' '}
              <M>{'\\lfloor n/d \\rfloor'}</M> independent directions per dimension, giving up to{' '}
              <M>{'\\lfloor n/d\\rfloor^{d}'}</M> copies of the downstream pattern. Each
              additional layer applies the same factor to whatever has already been built:
            </p>
            <Equation plain>
              {'\\#\\text{regions} \\;\\ge\\; \\left\\lfloor \\frac{n}{d}\\right\\rfloor^{d(L-1)} \\sum_{j=0}^{d}\\binom{n}{j}'}
            </Equation>
            <InWords tag="reading it">
              <p>
                <M>{'\\lfloor \\cdot \\rfloor'}</M> means round down to a whole number. The
                right-hand factor is the single-layer count from above; the left-hand factor is what
                depth multiplies it by.
              </p>
              <p>
                The important feature is where <M>{'L'}</M> sits: in the <em>exponent</em>. Adding a
                layer multiplies the count, while adding units to an existing layer only adds to it.
                Multiplying repeatedly is what produces exponential growth.
              </p>
            </InWords>
            <p>
              The base is the single-layer count and the prefactor is the compounding from depth.
              Parameters, by contrast, grow linearly in <M>{'L'}</M>: about{' '}
              <M>{'(L-1)n^2'}</M> of them. So the same budget of parameters buys polynomially many
              regions when spent on width and exponentially many when spent on depth.
            </p>
            <p>
              The table below evaluates both expressions. At <M>{'n = 8'}</M>, going from one hidden
              layer to four multiplies the parameters by about 7.5 and the region bound by about
              4000.
            </p>
          </Detail>
        </div>
      </div>

      <div className="grid grid--side">
        <Panel title="Region count, d = 2 inputs" flush>
          <div style={{ padding: '14px 16px 4px' }}>
            <Slider
              label="Width n"
              min={2}
              max={20}
              step={1}
              value={counterWidth}
              onChange={setCounterWidth}
              display={counterWidth}
            />
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>Hidden layers</th>
                <th>Parameters</th>
                <th>Linear regions</th>
              </tr>
            </thead>
            <tbody>
              {[1, 2, 3, 4].map((layers) => {
                const n = counterWidth;
                const params =
                  2 * n + n + (layers - 1) * (n * n + n) + n + 1;
                const regions = layers === 1 ? shallowRegions(n) : deepRegions(n, layers);
                return (
                  <tr key={layers}>
                    <td>{layers}</td>
                    <td>{params}</td>
                    <td>{regions.toLocaleString('en-US')}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="figure__caption">
            Row 1 is the exact maximum for one hidden layer. Rows 2–4 are lower bounds from the
            standard construction: each additional layer contributes a factor of ⌊n/d⌋ᵈ. Parameter
            count grows linearly in depth while the region count grows exponentially.
          </div>
        </Panel>

        <div className="prose-block">
          <h3 className="subhead">What depth does not give you</h3>
          <p>
            The universal approximation theorem is precise and worth stating exactly, because it is
            often quoted loosely. For any continuous <M>{'h'}</M> on a compact set{' '}
            <M>{'K \\subset \\mathbb{R}^{d}'}</M>, any non-polynomial activation{' '}
            <M>{'f'}</M>, and any <M>{'\\epsilon > 0'}</M>, there exists a width{' '}
            <M>{'n'}</M> and parameters such that the one-hidden-layer network{' '}
            <M>{'\\hat{h}'}</M> satisfies{' '}
            <M>{'\\sup_{\\mathbf{x}\\in K}|h(\\mathbf{x}) - \\hat{h}(\\mathbf{x})| < \\epsilon'}</M>.
          </p>
          <p>
            Note what it does not say. It gives no bound on <M>{'n'}</M> — which can be exponential
            in <M>{'d'}</M> and in <M>{'1/\\epsilon'}</M>. It says nothing about whether gradient
            descent can find those parameters. And it says nothing about generalisation: it is a
            statement about fitting a known function on a compact set, not about learning one from
            samples.
          </p>
          <p>
            So depth is not about what is representable in principle — it is about how many units are
            required, and how findable the parameters are. Depth-separation results make the first
            part concrete: there are functions computable exactly by a network of depth{' '}
            <M>{'L'}</M> and polynomial width that require width exponential in{' '}
            <M>{'L'}</M> to approximate with depth <M>{'L-1'}</M>.
          </p>

          <h3 className="subhead">What depth costs</h3>
          <ul>
            <li>
              <strong>Gradient conditioning.</strong> Each layer adds a factor of{' '}
              <M>{"f'(z)"}</M> and <M>{'W^{\\top}'}</M> to the backward product, as section 08
              derived. Deeper means more factors and a product further from 1.
            </li>
            <li>
              <strong>Optimisation difficulty.</strong> More layers means a more composed, less
              well-conditioned surface, so the effective condition number of section 07 rises.
            </li>
            <li>
              <strong>Memory.</strong> Activations from every layer are held for the backward pass,
              so memory grows linearly in depth at fixed width.
            </li>
            <li>
              <strong>Latency.</strong> Layers are sequential. Width parallelises across units;
              depth does not.
            </li>
          </ul>
          <p>
            Set the activation above to sigmoid and train. The three-layer network now learns more
            slowly than the one-layer network despite being strictly more expressive — the gradient
            factor <M>{"\\sigma' \\le 0.25"}</M> costs more than the extra capacity gains. The
            techniques that made very deep networks practical — ReLU, careful initialisation,
            normalisation, residual connections — all address this list rather than expressiveness.
          </p>
        </div>
      </div>

      <Stats
        items={trainer.results.map((result) => ({
          label: `${result.network.sizes.length - 1}L loss`,
          value: fmt(result.loss, 3),
        }))}
      />

      <Note title="Reading the comparison honestly" accent>
        <p>
          At equal epochs the deeper networks usually reach a lower loss on Spiral and Circles and
          make no difference on Linear, where a single boundary is already optimal. The comparison
          holds only for this training budget: given enough units and epochs the shallow network can
          also fit Spiral. The claim depth supports is about efficiency — fewer parameters and fewer
          epochs for the same fit — not about possibility.
        </p>
      </Note>
    </Section>
  );
}
