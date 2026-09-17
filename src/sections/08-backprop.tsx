import { useMemo, useRef, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { useMutableNetwork } from '../hooks/useMutableNetwork.ts';
import { ACTIVATIONS } from '../lib/activations.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Button, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { fmt } from '../lib/format.ts';

const CONFIG: NetworkConfig = {
  inputSize: 2,
  hiddenUnits: [2],
  outputSize: 1,
  hiddenActivation: 'tanh',
  outputActivation: 'sigmoid',
  loss: 'bce',
  seed: 2024,
};

const PRESET = {
  W: [
    [
      [0.8, -0.3],
      [0.4, 0.9],
    ],
    [[1.2, -0.7]],
  ],
  b: [[0.2, -0.1], [0.15]],
};

interface BackStep {
  title: string;
  quantity: string;
  detail: string;
}

const STEPS: BackStep[] = [
  {
    title: 'Forward pass',
    quantity: 'z⁽¹⁾, a⁽¹⁾, z⁽²⁾, ŷ',
    detail: 'Backpropagation needs every activation and pre-activation, so they are stored during the forward pass.',
  },
  {
    title: '∂L/∂ŷ',
    quantity: '∂L/∂ŷ',
    detail: 'The derivative of the loss with respect to the network output. This is where the backward pass starts.',
  },
  {
    title: 'δ⁽²⁾ = ∂L/∂z⁽²⁾',
    quantity: 'δ⁽²⁾',
    detail: 'Multiply by the output activation’s derivative. For sigmoid with binary cross-entropy the product simplifies to ŷ − y.',
  },
  {
    title: '∂L/∂W⁽²⁾ and ∂L/∂b⁽²⁾',
    quantity: '∂L/∂W⁽²⁾',
    detail: 'Since z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾, the derivative with respect to a weight is δ⁽²⁾ times the activation it multiplied.',
  },
  {
    title: '∂L/∂a⁽¹⁾',
    quantity: '∂L/∂a⁽¹⁾',
    detail: 'Each hidden activation influenced the output through its outgoing weights, so its gradient is the weighted sum of the δ values it fed.',
  },
  {
    title: 'δ⁽¹⁾ = ∂L/∂z⁽¹⁾',
    quantity: 'δ⁽¹⁾',
    detail: 'Multiply the activation gradient by f′(z⁽¹⁾) elementwise. This is the only place the hidden activation’s shape enters.',
  },
  {
    title: '∂L/∂W⁽¹⁾ and ∂L/∂b⁽¹⁾',
    quantity: '∂L/∂W⁽¹⁾',
    detail: 'Same rule as layer 2, one layer earlier: the gradient of a weight is δ for its unit times the input it multiplied.',
  },
  {
    title: 'Apply the update',
    quantity: 'W ← W − η ∂L/∂W',
    detail: 'One gradient-descent step on every parameter. The loss on this example should decrease for a small enough η.',
  },
];

export function BackpropSection({ id, index }: SectionProps) {
  const config = useMemo(() => CONFIG, []);
  const { network, version, mutate } = useMutableNetwork(config);
  const presetApplied = useRef(false);
  const [step, setStep] = useState(0);
  const [x, setX] = useState<[number, number]>([1.0, 0.5]);
  const [target, setTarget] = useState(1);
  const [learningRate, setLearningRate] = useState(0.5);
  const [selected, setSelected] = useState<{ layer: number; to: number; from: number }>({
    layer: 0,
    to: 0,
    from: 0,
  });
  const [numericCheck, setNumericCheck] = useState<number | null>(null);

  // Fixed starting weights, so the worked numbers below are stable and the
  // chain-rule arithmetic can be checked by hand.
  if (!presetApplied.current) {
    presetApplied.current = true;
    network.W = PRESET.W.map((layer) => layer.map((row) => row.slice()));
    network.b = PRESET.b.map((row) => row.slice());
  }

  const trace = useMemo(
    () => network.forward(x),
    // `version` is the dependency that matters: the network is mutated in place.
    [network, x, version],
  );
  const grads = useMemo(
    () => network.backward(trace, [target]),
    // `version` is the dependency that matters: the network is mutated in place.
    [network, trace, target, version],
  );

  const loss = network.sampleLoss(trace.output, [target]);
  const yHat = trace.output[0];
  const a1 = trace.layers[0].a;
  const z1 = trace.layers[0].z;
  const z2 = trace.layers[1].z[0];
  const delta2 = grads.delta[1][0];
  const dA1 = grads.dA[0];
  const delta1 = grads.delta[0];
  const hiddenAct = ACTIVATIONS[config.hiddenActivation];

  const selectedGrad = grads.dW[selected.layer][selected.to][selected.from];
  const selectedWeight = network.W[selected.layer][selected.to][selected.from];

  const verify = () => {
    const h = 1e-5;
    const original = network.W[selected.layer][selected.to][selected.from];
    network.W[selected.layer][selected.to][selected.from] = original + h;
    const up = network.sampleLoss(network.forward(x).output, [target]);
    network.W[selected.layer][selected.to][selected.from] = original - h;
    const down = network.sampleLoss(network.forward(x).output, [target]);
    network.W[selected.layer][selected.to][selected.from] = original;
    setNumericCheck((up - down) / (2 * h));
  };

  const applyUpdate = () => {
    mutate((net) => {
      for (let l = 0; l < net.W.length; l++) {
        for (let j = 0; j < net.W[l].length; j++) {
          net.b[l][j] -= learningRate * grads.db[l][j];
          for (let i = 0; i < net.W[l][j].length; i++) {
            net.W[l][j][i] -= learningRate * grads.dW[l][j][i];
          }
        }
      }
    });
    setNumericCheck(null);
  };

  const chainLatex =
    selected.layer === 1
      ? '\\frac{\\partial L}{\\partial W^{(2)}_{1i}} = \\underbrace{\\frac{\\partial L}{\\partial \\hat{y}}}_{\\text{loss}} \\cdot \\underbrace{\\frac{\\partial \\hat{y}}{\\partial z^{(2)}}}_{\\sigma\'(z^{(2)})} \\cdot \\underbrace{\\frac{\\partial z^{(2)}}{\\partial W^{(2)}_{1i}}}_{a^{(1)}_i}'
      : '\\frac{\\partial L}{\\partial W^{(1)}_{ji}} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial z^{(2)}} \\cdot \\frac{\\partial z^{(2)}}{\\partial a^{(1)}_j} \\cdot \\frac{\\partial a^{(1)}_j}{\\partial z^{(1)}_j} \\cdot \\frac{\\partial z^{(1)}_j}{\\partial W^{(1)}_{ji}}';

  const chainNumbers =
    selected.layer === 1
      ? [
          { label: '∂L/∂ŷ', value: (yHat - target) / (yHat * (1 - yHat)) },
          { label: "σ'(z⁽²⁾)", value: yHat * (1 - yHat) },
          { label: `a⁽¹⁾${selected.from + 1}`, value: a1[selected.from] },
        ]
      : [
          { label: '∂L/∂ŷ', value: (yHat - target) / (yHat * (1 - yHat)) },
          { label: "σ'(z⁽²⁾)", value: yHat * (1 - yHat) },
          { label: `w⁽²⁾${selected.to + 1}`, value: network.W[1][0][selected.to] },
          { label: `f'(z⁽¹⁾${selected.to + 1})`, value: hiddenAct.df(z1[selected.to]) },
          { label: `x${selected.from + 1}`, value: x[selected.from] },
        ];
  const chainProduct = chainNumbers.reduce((acc, item) => acc * item.value, 1);

  const revealBackward = step >= 1;

  return (
    <Section
      id={id}
      index={index}
      title="Backpropagation"
      lede={
        <>
          Backpropagation computes <M>{'\\partial L / \\partial W'}</M> and{' '}
          <M>{'\\partial L / \\partial b'}</M> for every parameter in one backward sweep. It is the
          chain rule applied to the composition of layers, evaluated from the output towards the
          input so that each intermediate result is used rather than recomputed.
        </>
      }
    >
      <Equation caption="δ is the gradient with respect to a layer's pre-activations. Everything else follows from it.">
        {'\\delta^{(L)} = \\frac{\\partial L}{\\partial \\mathbf{z}^{(L)}}, \\qquad \\delta^{(l)} = \\left(W^{(l+1)}\\right)^{\\!\\top}\\delta^{(l+1)} \\odot f\'\\!\\left(\\mathbf{z}^{(l)}\\right)'}
      </Equation>
      <Equation caption="The parameter gradients are outer products of δ with the incoming activations.">
        {'\\frac{\\partial L}{\\partial W^{(l)}} = \\delta^{(l)} \\left(\\mathbf{a}^{(l-1)}\\right)^{\\!\\top}, \\qquad \\frac{\\partial L}{\\partial \\mathbf{b}^{(l)}} = \\delta^{(l)}'}
      </Equation>

      <div className="grid grid--side">
        <Panel
          title={`Step ${step} — ${STEPS[step].title}`}
          hint={step === 0 ? 'forward' : step === 7 ? 'update' : 'backward'}
          caption={STEPS[step].detail}
        >
          <NetworkDiagram
            network={network}
            trace={trace}
            gradients={revealBackward ? grads : null}
            showGradients={revealBackward && step < 7}
            direction={revealBackward && step < 7 ? 'backward' : 'forward'}
            activeLayer={step === 3 || step === 4 ? 1 : step === 6 ? 0 : step === 0 ? 0 : null}
            height={250}
            inputLabels={['x₁', 'x₂']}
            outputLabels={['ŷ']}
            selection={{ kind: 'edge', layer: selected.layer, from: selected.from, to: selected.to }}
            onSelect={(selection) => {
              if (selection?.kind === 'edge') {
                setSelected({ layer: selection.layer, from: selection.from, to: selection.to });
                setNumericCheck(null);
              }
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
              onClick={() => {
                if (step === STEPS.length - 1) {
                  applyUpdate();
                  setStep(0);
                } else {
                  setStep((s) => s + 1);
                }
              }}
            >
              {step === STEPS.length - 1 ? 'Apply update and restart' : 'Next step →'}
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
                title={s.title}
              >
                {i}
              </button>
            ))}
          </div>
          <p className="faint" style={{ fontSize: 12.5, marginTop: 10, marginBottom: 0 }}>
            {revealBackward && step < 7
              ? 'Node shading now shows δ and edge width shows |∂L/∂w|; edges point backwards.'
              : 'Node shading shows activations and edge width shows |w|.'}
          </p>
        </Panel>

        <div className="stack">
          <Panel title="Example and hyperparameter">
            <div className="stack stack--sm">
              <Slider label="x₁" min={-2} max={2} value={x[0]} onChange={(v) => setX([v, x[1]])} display={fmt(x[0])} />
              <Slider label="x₂" min={-2} max={2} value={x[1]} onChange={(v) => setX([x[0], v])} display={fmt(x[1])} />
              <Segmented
                label="Target y"
                value={String(target)}
                options={[
                  { value: '1', label: 'y = 1' },
                  { value: '0', label: 'y = 0' },
                ]}
                onChange={(v) => setTarget(Number(v))}
              />
              <Slider
                label="Learning rate η"
                min={0.01}
                max={2}
                step={0.01}
                value={learningRate}
                onChange={setLearningRate}
                display={fmt(learningRate, 2)}
              />
            </div>
            <div className="btn-row" style={{ marginTop: 12 }}>
              <Button onClick={applyUpdate}>Apply one update</Button>
              <Button
                onClick={() => {
                  mutate((net) => {
                    net.W = PRESET.W.map((layer) => layer.map((row) => row.slice()));
                    net.b = PRESET.b.map((row) => row.slice());
                  });
                  setStep(0);
                  setNumericCheck(null);
                }}
              >
                Restore weights
              </Button>
            </div>
          </Panel>

          <Stats
            items={[
              { label: 'ŷ', value: fmt(yHat, 4) },
              { label: 'Loss', value: fmt(loss, 4), accent: true },
              { label: 'δ⁽²⁾', value: step >= 2 ? fmt(delta2, 4) : '—' },
              { label: 'δ⁽¹⁾₁', value: step >= 5 ? fmt(delta1[0], 4) : '—' },
              { label: 'δ⁽¹⁾₂', value: step >= 5 ? fmt(delta1[1], 4) : '—' },
            ]}
          />
        </div>
      </div>

      <div className="grid grid--side">
        <Panel title="Backward pass, with numbers" hint="values appear as you step">
          <div className="calc">
            <div className="calc__line">
              <span className="calc__label">forward   </span>
              z⁽¹⁾ = [{z1.map((v) => fmt(v, 4)).join(', ')}] a⁽¹⁾ = [{a1.map((v) => fmt(v, 4)).join(', ')}]
            </div>
            <div className="calc__line">
              <span className="calc__label">          </span>
              z⁽²⁾ = {fmt(z2, 4)} ŷ = σ(z⁽²⁾) = {fmt(yHat, 4)} L = {fmt(loss, 4)}
            </div>
            <div className="calc__line" style={{ opacity: step >= 1 ? 1 : 0.25 }}>
              <span className="calc__label">∂L/∂ŷ     </span>= (ŷ − y) / (ŷ(1−ŷ)) ={' '}
              {fmt((yHat - target) / (yHat * (1 - yHat)), 4)}
            </div>
            <div className="calc__line" style={{ opacity: step >= 2 ? 1 : 0.25 }}>
              <span className="calc__label">δ⁽²⁾      </span>= ∂L/∂ŷ · σ′(z⁽²⁾) = ŷ − y ={' '}
              <span className="calc__result">{fmt(delta2, 4)}</span>
            </div>
            <div className="calc__line" style={{ opacity: step >= 3 ? 1 : 0.25 }}>
              <span className="calc__label">∂L/∂W⁽²⁾  </span>= δ⁽²⁾ · a⁽¹⁾ = [
              {grads.dW[1][0].map((v) => fmt(v, 4)).join(', ')}]
            </div>
            <div className="calc__line" style={{ opacity: step >= 3 ? 1 : 0.25 }}>
              <span className="calc__label">∂L/∂b⁽²⁾  </span>= δ⁽²⁾ = {fmt(grads.db[1][0], 4)}
            </div>
            <div className="calc__line" style={{ opacity: step >= 4 ? 1 : 0.25 }}>
              <span className="calc__label">∂L/∂a⁽¹⁾  </span>= (W⁽²⁾)ᵀ δ⁽²⁾ = [
              {dA1.map((v) => fmt(v, 4)).join(', ')}]
            </div>
            <div className="calc__line" style={{ opacity: step >= 5 ? 1 : 0.25 }}>
              <span className="calc__label">δ⁽¹⁾      </span>= ∂L/∂a⁽¹⁾ ⊙ f′(z⁽¹⁾) = [
              {dA1.map((v) => fmt(v, 3)).join(', ')}] ⊙ [
              {z1.map((v) => fmt(hiddenAct.df(v), 3)).join(', ')}] = [
              {delta1.map((v) => fmt(v, 4)).join(', ')}]
            </div>
            <div className="calc__line" style={{ opacity: step >= 6 ? 1 : 0.25 }}>
              <span className="calc__label">∂L/∂W⁽¹⁾  </span>= δ⁽¹⁾ xᵀ = [[
              {grads.dW[0][0].map((v) => fmt(v, 4)).join(', ')}], [
              {grads.dW[0][1].map((v) => fmt(v, 4)).join(', ')}]]
            </div>
            <div className="calc__line" style={{ opacity: step >= 6 ? 1 : 0.25 }}>
              <span className="calc__label">∂L/∂b⁽¹⁾  </span>= δ⁽¹⁾ = [
              {grads.db[0].map((v) => fmt(v, 4)).join(', ')}]
            </div>
          </div>
        </Panel>

        <div className="stack">
          <Panel
            title={`Chain rule for w⁽${selected.layer + 1}⁾ ${selected.to + 1},${selected.from + 1}`}
            hint="click any edge in the diagram"
          >
            <Equation plain>{chainLatex}</Equation>
            <div className="calc">
              <div className="calc__line">
                {chainNumbers.map((item) => item.label).join('  ×  ')}
              </div>
              <div className="calc__line">
                {chainNumbers.map((item) => fmt(item.value, 4)).join('  ×  ')}
              </div>
              <div className="calc__line">
                = <span className="calc__result">{fmt(chainProduct, 6)}</span>
              </div>
              <div className="calc__line">
                <span className="calc__label">backprop gives </span>
                {fmt(selectedGrad, 6)}
              </div>
              {numericCheck !== null ? (
                <div className="calc__line">
                  <span className="calc__label">finite diff.   </span>
                  {fmt(numericCheck, 6)}
                  <span className="calc__label">
                    {'  '}|difference| = {fmt(Math.abs(numericCheck - selectedGrad), 9)}
                  </span>
                </div>
              ) : null}
            </div>
            <div className="btn-row" style={{ marginTop: 12 }}>
              <Button onClick={verify}>Check against finite differences</Button>
            </div>
            <p className="faint" style={{ fontSize: 12.5, marginTop: 10, marginBottom: 0 }}>
              The check evaluates <span className="mono">(L(w + h) − L(w − h)) / 2h</span> with{' '}
              <span className="mono">h = 10⁻⁵</span>, running two extra forward passes. Agreement to
              five or six decimals confirms the analytic derivative.
            </p>
          </Panel>

          <Panel title="Current value of the selected weight">
            <Stats
              items={[
                { label: 'w', value: fmt(selectedWeight, 4) },
                { label: '∂L/∂w', value: fmt(selectedGrad, 4), accent: true },
                { label: 'w after update', value: fmt(selectedWeight - learningRate * selectedGrad, 4) },
              ]}
            />
          </Panel>
        </div>
      </div>

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">Why work backwards</h3>
          <p>
            A forward-mode derivative computes <M>{'\\partial \\hat{y} / \\partial \\theta_k'}</M>{' '}
            for one parameter at a time, requiring one pass per parameter. Reverse mode computes{' '}
            <M>{'\\partial L / \\partial \\theta_k'}</M> for all parameters in a single pass,
            because the loss is a single scalar and the intermediate{' '}
            <M>{'\\delta'}</M> values are shared by every weight in a layer.
          </p>
          <p>
            For a network with <M>{'P'}</M> parameters, forward mode costs <M>{'O(P)'}</M> passes
            and reverse mode costs <M>{'O(1)'}</M>. With millions of parameters this difference is
            what makes training feasible at all.
          </p>
        </div>
        <div className="prose-block">
          <h3 className="subhead">Where gradients vanish</h3>
          <p>
            Each backward step through a layer multiplies by <M>{"f'(z)"}</M> and by{' '}
            <M>{'W^{\\top}'}</M>. Tracing a path from the loss back to an early layer produces a
            product of many such factors. If the typical factor is below 1 the product decays
            geometrically with depth; if it is above 1 the product explodes.
          </p>
          <p>
            Set the hidden activation to sigmoid in section 05 and note that{' '}
            <M>{"\\sigma'(z) \\le 0.25"}</M>. Careful initialisation, ReLU-family activations,
            normalisation layers and residual connections all exist to keep this product near 1.
          </p>
        </div>
      </div>

      <Note title="An identity worth memorising" accent>
        <p>
          For sigmoid output with binary cross-entropy, and for softmax output with categorical
          cross-entropy, the first two factors collapse:{' '}
          <M>{'\\delta^{(L)} = \\hat{\\mathbf{y}} - \\mathbf{y}'}</M>. No division, no saturating
          factor. This is why frameworks expose a single fused operation
          (<code>softmax_cross_entropy_with_logits</code>) rather than composing the two.
        </p>
      </Note>
    </Section>
  );
}
