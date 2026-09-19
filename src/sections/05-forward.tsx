import { useEffect, useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig, ForwardTrace } from '../lib/network.ts';
import { useMutableNetwork } from '../hooks/useMutableNetwork.ts';
import { ACTIVATIONS, HIDDEN_ACTIVATIONS } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Detail, InWords, Steps } from '../components/ui/Detail.tsx';
import { Button, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Exercise } from '../components/ui/Exercise.tsx';
import { fmt } from '../lib/format.ts';

interface Step {
  title: string;
  /** Weight-layer index being used, for edge animation. */
  layer: number | null;
  /** How many layers of the trace to reveal. */
  reveal: number;
  /** Show pre-activations rather than activations for the last revealed layer. */
  preActivation: boolean;
}

const STEPS: Step[] = [
  { title: 'Inputs', layer: null, reveal: 0, preActivation: false },
  { title: 'Hidden pre-activations z⁽¹⁾', layer: 0, reveal: 1, preActivation: true },
  { title: 'Hidden activations a⁽¹⁾ = f(z⁽¹⁾)', layer: null, reveal: 1, preActivation: false },
  { title: 'Output pre-activation z⁽²⁾', layer: 1, reveal: 2, preActivation: true },
  { title: 'Prediction ŷ = σ(z⁽²⁾)', layer: null, reveal: 2, preActivation: false },
];

function matrixLatex(rows: number[][], digits = 2): string {
  return `\\begin{bmatrix} ${rows
    .map((row) => row.map((v) => fmt(v, digits).replace('−', '-')).join(' & '))
    .join(' \\\\ ')} \\end{bmatrix}`;
}

function columnLatex(values: number[], digits = 2): string {
  return matrixLatex(values.map((v) => [v]), digits);
}

/** A column of question marks, shown for results the reader has not stepped to yet. */
const PLACEHOLDER_COLUMN = (rows: number): string =>
  `\\begin{bmatrix} ${Array.from({ length: rows }, () => '?').join(' \\\\ ')} \\end{bmatrix}`;

export function ForwardSection({ id, index }: SectionProps) {
  const [hiddenActivation, setHiddenActivation] = useState<ActivationName>('relu');
  const config = useMemo<NetworkConfig>(
    () => ({
      inputSize: 2,
      hiddenUnits: [3],
      outputSize: 1,
      hiddenActivation,
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 314,
    }),
    [hiddenActivation],
  );
  const { network, version, reset } = useMutableNetwork(config);
  const [x, setX] = useState<[number, number]>([1.0, 0.5]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [unit, setUnit] = useState(0);

  const trace = useMemo(
    () => network.forward(x),
    // `version` is the dependency that matters: the network is mutated in place.
    [network, x, version],
  );

  useEffect(() => {
    if (!playing) return;
    const handle = window.setTimeout(() => {
      setStep((current) => {
        if (current >= STEPS.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1100);
    return () => window.clearTimeout(handle);
  }, [playing, step]);

  const current = STEPS[step];

  /** A trace truncated to the revealed layers, with z substituted where needed. */
  const displayTrace = useMemo<ForwardTrace>(() => {
    const layers = trace.layers.slice(0, current.reveal).map((layer, i) => {
      const isLast = i === current.reveal - 1;
      return isLast && current.preActivation ? { ...layer, a: layer.z } : layer;
    });
    return { input: trace.input, layers, output: layers[layers.length - 1]?.a ?? [] };
  }, [trace, current]);

  const hiddenUnits = network.sizes[1];
  const selectedUnit = Math.min(unit, hiddenUnits - 1);
  const act = ACTIVATIONS[hiddenActivation];

  const wRow = network.W[0][selectedUnit];
  const bValue = network.b[0][selectedUnit];
  const zValue = trace.layers[0].z[selectedUnit];
  const aValue = trace.layers[0].a[selectedUnit];

  const outW = network.W[1][0];
  const outB = network.b[1][0];
  const outZ = trace.layers[1].z[0];
  const yHat = trace.output[0];

  const showOutputUnit = step >= 3;

  return (
    <Section
      id={id}
      index={index}
      title="Forward propagation"
      lede={
        <>
          Forward propagation evaluates the network on an input: each layer computes its
          pre-activations from the previous layer's activations, applies <M>{'f'}</M>, and passes
          the result on. Nothing is learned during a forward pass — it is pure evaluation.
        </>
      }
    >
      <div className="grid grid--side">
        <Panel
          title={`Step ${step} of ${STEPS.length - 1} — ${current.title}`}
          hint={step === 0 ? 'the input vector' : current.layer !== null ? 'multiplying and summing' : 'applying f'}
          caption="Values appear only once they have been computed. Animated edges are the products being accumulated in the current step."
        >
          <NetworkDiagram
            network={network}
            trace={displayTrace}
            activeLayer={current.layer}
            height={270}
            inputLabels={['x₁', 'x₂']}
            outputLabels={['ŷ']}
            selection={step >= 1 && step <= 2 ? { kind: 'node', layer: 1, index: selectedUnit } : null}
            onSelect={(selection) => {
              if (selection?.kind === 'node' && selection.layer === 1) setUnit(selection.index);
            }}
          />
          <div className="btn-row" style={{ marginTop: 8 }}>
            <Button onClick={() => setStep(0)} disabled={step === 0}>
              Reset
            </Button>
            <Button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
              ← Previous
            </Button>
            <Button
              variant="primary"
              onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}
              disabled={step === STEPS.length - 1}
            >
              Next step →
            </Button>
            <Button
              onClick={() => {
                if (step === STEPS.length - 1) setStep(0);
                setPlaying(!playing);
              }}
            >
              {playing ? 'Pause' : 'Run all'}
            </Button>
          </div>
          <div className="row" style={{ marginTop: 10, gap: 6 }}>
            {STEPS.map((s, i) => (
              <button
                key={s.title}
                type="button"
                className={i === step ? 'segmented__option segmented__option--active' : 'segmented__option'}
                onClick={() => setStep(i)}
                style={{ fontSize: 11.5 }}
              >
                {i}
              </button>
            ))}
            <span className="faint" style={{ fontSize: 12 }}>
              {current.title}
            </span>
          </div>
        </Panel>

        <div className="stack">
          <Panel title="Input and settings">
            <div className="stack stack--sm">
              <Slider label="x₁" min={-2} max={2} value={x[0]} onChange={(v) => setX([v, x[1]])} display={fmt(x[0])} />
              <Slider label="x₂" min={-2} max={2} value={x[1]} onChange={(v) => setX([x[0], v])} display={fmt(x[1])} />
              <Segmented
                label="Hidden activation f"
                value={hiddenActivation}
                options={HIDDEN_ACTIVATIONS.map((n) => ({ value: n, label: ACTIVATIONS[n].label }))}
                onChange={setHiddenActivation}
              />
              <div className="btn-row">
                <Button onClick={() => reset(Math.floor(Math.random() * 100000))}>New weights</Button>
              </div>
            </div>
          </Panel>

          <Stats
            items={[
              { label: 'z⁽¹⁾ selected', value: step >= 1 ? fmt(zValue, 3) : '—' },
              { label: 'a⁽¹⁾ selected', value: step >= 2 ? fmt(aValue, 3) : '—' },
              { label: 'z⁽²⁾', value: step >= 3 ? fmt(outZ, 3) : '—' },
              { label: 'ŷ', value: step >= 4 ? fmt(yHat, 3) : '—', accent: true },
            ]}
          />
        </div>
      </div>

      <div className="grid grid--side">
        <Panel
          title={showOutputUnit ? 'Output unit' : `Hidden unit ${selectedUnit + 1}`}
          hint="click a hidden unit in the diagram to switch"
        >
          <div className="calc">
            {showOutputUnit ? (
              <>
                <div className="calc__line">
                  <span className="calc__label">inputs      </span>
                  a⁽¹⁾ = [{trace.layers[0].a.map((v) => fmt(v, 3)).join(', ')}]
                </div>
                <div className="calc__line">
                  <span className="calc__label">weights     </span>
                  w⁽²⁾ = [{outW.map((v) => fmt(v, 3)).join(', ')}]
                </div>
                <div className="calc__line">
                  <span className="calc__label">weighted sum</span>{' '}
                  {outW.map((w, i) => `(${fmt(w, 2)})(${fmt(trace.layers[0].a[i], 2)})`).join(' + ')}
                </div>
                <div className="calc__line">
                  <span className="calc__label">            </span>= {fmt(outZ - outB, 4)}
                </div>
                <div className="calc__line">
                  <span className="calc__label">+ bias      </span>
                  {fmt(outZ - outB, 4)} + {fmt(outB, 3)} ={' '}
                  <span className="calc__result">{fmt(outZ, 4)}</span>
                </div>
                <div className="calc__line">
                  <span className="calc__label">activation  </span>
                  σ({fmt(outZ, 4)}) = 1 / (1 + e^{fmt(-outZ, 4)})
                </div>
                <div className="calc__line">
                  <span className="calc__label">output      </span>
                  ŷ = <span className="calc__result">{step >= 4 ? fmt(yHat, 4) : '…'}</span>
                </div>
              </>
            ) : (
              <>
                <div className="calc__line">
                  <span className="calc__label">inputs      </span>x = [{fmt(x[0], 2)}, {fmt(x[1], 2)}]
                </div>
                <div className="calc__line">
                  <span className="calc__label">weights     </span>w = [{wRow.map((v) => fmt(v, 3)).join(', ')}]
                </div>
                <div className="calc__line">
                  <span className="calc__label">weighted sum</span>{' '}
                  ({fmt(wRow[0], 2)})({fmt(x[0], 2)}) + ({fmt(wRow[1], 2)})({fmt(x[1], 2)})
                </div>
                <div className="calc__line">
                  <span className="calc__label">            </span>={' '}
                  {step >= 1
                    ? `${fmt(wRow[0] * x[0], 4)} + ${fmt(wRow[1] * x[1], 4)} = ${fmt(zValue - bValue, 4)}`
                    : '…'}
                </div>
                <div className="calc__line">
                  <span className="calc__label">+ bias      </span>
                  {step >= 1 ? `${fmt(zValue - bValue, 4)} + ${fmt(bValue, 3)} = ` : `${fmt(bValue, 3)} → `}
                  <span className={step >= 1 ? 'calc__result' : ''}>{step >= 1 ? fmt(zValue, 4) : '…'}</span>
                </div>
                <div className="calc__line">
                  <span className="calc__label">activation  </span>
                  {act.label}({step >= 1 ? fmt(zValue, 4) : 'z'}) ={' '}
                  <span className={step >= 2 ? 'calc__result' : ''}>{step >= 2 ? fmt(aValue, 4) : '…'}</span>
                </div>
              </>
            )}
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            {Array.from({ length: hiddenUnits }, (_, i) => (
              <button
                key={i}
                type="button"
                className={
                  i === selectedUnit && !showOutputUnit
                    ? 'segmented__option segmented__option--active'
                    : 'segmented__option'
                }
                onClick={() => {
                  setUnit(i);
                  if (step > 2) setStep(2);
                }}
              >
                hidden {i + 1}
              </button>
            ))}
          </div>
        </Panel>

        <div className="stack">
          <Panel title="The same step in matrix form" hint="fills in as you step">
            <div style={{ marginBottom: 12 }}>
              <InWords tag="reminder">
                <p>
                  The grid is <M>{'W^{(1)}'}</M>: one row per hidden unit, one column per input. The
                  column beside it is the input. Multiplying them runs the sum-of-products once per
                  row, exactly as in section 00, and the column on the right holds the three results.
                </p>
              </InWords>
            </div>
            <Equation plain>
              {`\\mathbf{z}^{(1)} = ${matrixLatex(network.W[0], 2)} ${columnLatex(x, 2)} + ${columnLatex(network.b[0], 2)} = ${step >= 1 ? columnLatex(trace.layers[0].z, 3) : PLACEHOLDER_COLUMN(3)}`}
            </Equation>
            <Equation plain>
              {`\\mathbf{a}^{(1)} = f\\!\\left(\\mathbf{z}^{(1)}\\right) = ${step >= 2 ? columnLatex(trace.layers[0].a, 3) : PLACEHOLDER_COLUMN(3)}`}
            </Equation>
            <Equation plain>
              {`z^{(2)} = ${matrixLatex([outW], 2)}\\,\\mathbf{a}^{(1)} + ${fmt(outB, 2).replace('−', '-')} = ${step >= 3 ? fmt(outZ, 4).replace('−', '-') : '\\;?\\;'}`}
            </Equation>
            <Equation plain>
              {`\\hat{y} = \\sigma\\!\\left(z^{(2)}\\right) = ${step >= 4 ? fmt(yHat, 4).replace('−', '-') : '\\;?\\;'}`}
            </Equation>
          </Panel>

          <div className="prose-block">
            <p>
              The loops in the panel on the left and the matrix product here are the same
              arithmetic. Frameworks use the matrix form because a single matrix multiplication
              evaluates every unit in the layer at once, and because a whole batch of inputs becomes
              one matrix product: stacking <M>{'m'}</M> examples as columns of{' '}
              <M>{'X \\in \\mathbb{R}^{n_0 \\times m}'}</M> gives{' '}
              <M>{'Z^{(1)} = W^{(1)}X + \\mathbf{b}^{(1)}'}</M>, with the bias added to every column.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">The algorithm, stated once</h3>
          <p>
            Forward propagation is four lines. Everything else on this page is a consequence of
            them.
          </p>
          <Steps>
            <li>
              Set <M>{'\\mathbf{a}^{(0)} = \\mathbf{x}'}</M>.
            </li>
            <li>
              For <M>{'l = 1, \\ldots, L'}</M>: compute{' '}
              <M>{'\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}'}</M>.
            </li>
            <li>
              Apply the activation: <M>{'\\mathbf{a}^{(l)} = f_l(\\mathbf{z}^{(l)})'}</M>, with{' '}
              <M>{'f_L'}</M> the output activation.
            </li>
            <li>
              Return <M>{'\\hat{\\mathbf{y}} = \\mathbf{a}^{(L)}'}</M>, and — if the result will
              be trained on — keep every <M>{'\\mathbf{z}^{(l)}'}</M> and{' '}
              <M>{'\\mathbf{a}^{(l)}'}</M>.
            </li>
          </Steps>
          <p>
            Step 4 is the one that is easy to miss. Backpropagation needs{' '}
            <M>{'\\mathbf{a}^{(l-1)}'}</M> to form <M>{'\\partial L/\\partial W^{(l)}'}</M> and{' '}
            <M>{'\\mathbf{z}^{(l)}'}</M> to evaluate <M>{"f'"}</M>. Recomputing them would double
            the work, so they are stored instead. This is the entire reason training uses more
            memory than inference, and the reason a batch that fits at inference time may not fit
            during training.
          </p>

          <Detail title="Memory held by the forward pass">
            <p>
              For a batch of <M>{'m'}</M> examples, layer <M>{'l'}</M> holds{' '}
              <M>{'m\\,n_l'}</M> pre-activations and <M>{'m\\,n_l'}</M> activations, so the total
              retained is <M>{'2m\\sum_{l} n_l'}</M> numbers, on top of the{' '}
              <M>{'\\sum_l (n_l n_{l-1} + n_l)'}</M> parameters and an equal number of gradient
              slots.
            </p>
            <p>
              Take a network with ten hidden layers of 1024 units, a batch of 128, and 32-bit
              floats. Activations: <M>{'2 \\times 128 \\times 10 \\times 1024 \\times 4'}</M>{' '}
              bytes <M>{'\\approx 10.5'}</M> MB. Parameters: about 9.4 million weights at 4 bytes
              each <M>{'\\approx 38'}</M> MB, plus the same again for the gradients. Doubling the
              batch doubles the activation term and leaves the parameter terms unchanged — which is
              why activation memory dominates at large batch sizes and parameter memory dominates at
              small ones.
            </p>
            <p>
              Gradient checkpointing trades one against the other: store only every{' '}
              <M>{'k'}</M>-th layer's activations and recompute the rest during the backward pass.
              Memory falls by roughly a factor of <M>{'k'}</M>, compute rises by roughly one extra
              forward pass.
            </p>
          </Detail>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Batching, concretely</h3>
          <p>
            With the network above (2 → 3 → 1) and a batch of four examples, the shapes are:
          </p>
          <Equation plain>
            {'\\underbrace{Z^{(1)}}_{3\\times 4} = \\underbrace{W^{(1)}}_{3\\times 2}\\underbrace{X}_{2\\times 4} + \\underbrace{\\mathbf{b}^{(1)}}_{3\\times 1}'}
          </Equation>
          <InWords>
            <p>
              The labels under each symbol are its shape — rows × columns. Four examples become four
              columns side by side, and the same weight grid processes all of them in one
              multiplication, producing a 3 × 4 grid: three hidden units for each of the four
              examples.
            </p>
          </InWords>
          <p>
            The bias has three entries but the result has twelve. The addition is a broadcast: the
            single column <M>{'\\mathbf{b}^{(1)}'}</M> is added to each of the four columns of the
            product. Written out, <M>{'Z^{(1)}_{jk} = \\sum_i W^{(1)}_{ji}X_{ik} + b^{(1)}_j'}</M> —
            the bias index does not involve <M>{'k'}</M>.
          </p>
          <p>
            The parameters are shared across the batch and the batch dimension is independent:
            column <M>{'k'}</M> of the output depends only on column <M>{'k'}</M> of the input. That
            is what makes batching a pure efficiency win. It changes nothing about the function,
            only how many examples are evaluated per matrix multiplication — and one large matrix
            product uses hardware far better than many small ones.
          </p>
          <p>
            Some layers break that independence deliberately. Batch normalisation computes each
            unit's mean and variance across the batch, so its output for one example depends on the
            others in the same batch. That is why such layers behave differently at training and
            evaluation time, and why the batch size becomes part of the model rather than only part
            of the optimiser.
          </p>

          <h3 className="subhead">Counting the work</h3>
          <p>
            Layer <M>{'l'}</M> performs one multiply and one add per weight, per example:{' '}
            <M>{'2\\,m\\,n_l n_{l-1}'}</M> floating-point operations, plus{' '}
            <M>{'m\\,n_l'}</M> for the bias and roughly <M>{'m\\,n_l'}</M> for the activation. The
            weight term dominates as soon as the layers are more than a few units wide, so the cost
            of a forward pass is <M>{'\\Theta(m \\sum_l n_l n_{l-1})'}</M> — the same order as the
            parameter count, times the batch size.
          </p>
          <p>
            For the 2 → 3 → 1 network above: <M>{'2\\cdot 3 = 6'}</M> multiply-adds in layer 1 and{' '}
            <M>{'3\\cdot 1 = 3'}</M> in layer 2, so 9 per example, about 18 FLOPs. Backpropagation
            costs roughly twice that, because it computes both{' '}
            <M>{'\\partial L/\\partial W'}</M> and the gradient passed to the previous layer.
          </p>
        </div>
      </div>

      <Note title="Forward propagation is deterministic — with two exceptions" accent>
        <p>
          Given the parameters and the input, the output is fixed. Two mechanisms break that
          deliberately, and both are active only during training: dropout multiplies hidden
          activations by a random mask, and batch normalisation makes the output depend on the other
          examples in the batch. Both switch off at evaluation time — dropout stops masking, batch
          normalisation uses running statistics instead of batch statistics.
        </p>
        <p>
          This is why frameworks have an explicit <code>train</code> / <code>eval</code> mode, and
          why forgetting to switch is a common source of a validation score that is quietly wrong.
          The forward pass in section 09 and beyond applies dropout when training and not when
          reporting metrics, for the same reason.
        </p>
      </Note>

      <Exercise
        notebook="05-forward.ipynb"
        count={2}
        tasks={[
          <>Write <code>layer_forward</code>, then <code>forward</code> with its cache</>,
          <>The check confirms a batch of 7 gives the same answer as 7 single rows</>,
          <>Measure the activation memory a batch holds, and why it — not the parameters — fills the GPU</>,
        ]}
      >
        Chain the layers into a forward pass, and store what backpropagation will need.
      </Exercise>
    </Section>
  );
}
