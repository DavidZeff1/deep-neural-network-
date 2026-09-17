import { useMemo, useRef, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { useMutableNetwork } from '../hooks/useMutableNetwork.ts';
import { ACTIVATIONS } from '../lib/activations.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Detail, Steps } from '../components/ui/Detail.tsx';
import { Button, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { fmt, sub } from '../lib/format.ts';

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
          { label: `a⁽¹⁾${sub(selected.from + 1)}`, value: a1[selected.from] },
        ]
      : [
          { label: '∂L/∂ŷ', value: (yHat - target) / (yHat * (1 - yHat)) },
          { label: "σ'(z⁽²⁾)", value: yHat * (1 - yHat) },
          { label: `w⁽²⁾${sub(selected.to + 1)}`, value: network.W[1][0][selected.to] },
          { label: `f′(z⁽¹⁾${sub(selected.to + 1)})`, value: hiddenAct.df(z1[selected.to]) },
          { label: `x${sub(selected.from + 1)}`, value: x[selected.from] },
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

      <div className="prose-block">
        <Detail kicker="derivation" title="Deriving the four equations from the chain rule">
          <p>
            Every quantity in the network is a function of the ones before it. The multivariable
            chain rule says that to differentiate through an intermediate vector, you sum over its
            components:
          </p>
          <Equation plain>
            {'\\frac{\\partial L}{\\partial u} = \\sum_{k} \\frac{\\partial L}{\\partial v_k}\\,\\frac{\\partial v_k}{\\partial u}'}
          </Equation>
          <p>
            Apply it four times. Write <M>{'\\delta^{(l)}_j = \\partial L/\\partial z^{(l)}_j'}</M>.
          </p>
          <p>
            <strong>1. The output layer.</strong> <M>{'z^{(L)}'}</M> affects <M>{'L'}</M> only
            through <M>{'\\hat{y} = g(z^{(L)})'}</M>, so
          </p>
          <Equation plain>
            {"\\delta^{(L)}_j = \\sum_k \\frac{\\partial L}{\\partial \\hat{y}_k}\\frac{\\partial \\hat{y}_k}{\\partial z^{(L)}_j}"}
          </Equation>
          <p>
            When <M>{'g'}</M> is elementwise the sum has one surviving term and this is{' '}
            <M>{"(\\partial L/\\partial \\hat{y}_j)\\,g'(z^{(L)}_j)"}</M>. When{' '}
            <M>{'g'}</M> is softmax every term survives, which is the Jacobian product.
          </p>
          <p>
            <strong>2. The recursion.</strong> <M>{'z^{(l)}_j'}</M> affects <M>{'L'}</M> only
            through <M>{'a^{(l)}_j = f(z^{(l)}_j)'}</M>, and that single activation feeds{' '}
            <em>every</em> unit of the next layer. So the sum runs over the next layer:
          </p>
          <Equation plain>
            {'\\delta^{(l)}_j = \\sum_p \\frac{\\partial L}{\\partial z^{(l+1)}_p}\\,\\frac{\\partial z^{(l+1)}_p}{\\partial z^{(l)}_j} = \\sum_p \\delta^{(l+1)}_p\\,\\frac{\\partial z^{(l+1)}_p}{\\partial z^{(l)}_j}'}
          </Equation>
          <p>
            The inner derivative is immediate from{' '}
            <M>{'z^{(l+1)}_p = \\sum_k W^{(l+1)}_{pk} f(z^{(l)}_k) + b^{(l+1)}_p'}</M>: only the{' '}
            <M>{'k = j'}</M> term depends on <M>{'z^{(l)}_j'}</M>, giving{' '}
            <M>{"W^{(l+1)}_{pj}f'(z^{(l)}_j)"}</M>. Substituting and pulling the common factor out of
            the sum:
          </p>
          <Equation plain>
            {"\\delta^{(l)}_j = f'(z^{(l)}_j)\\sum_p W^{(l+1)}_{pj}\\,\\delta^{(l+1)}_p"}
          </Equation>
          <p>
            The sum <M>{'\\sum_p W^{(l+1)}_{pj}\\delta^{(l+1)}_p'}</M> is the{' '}
            <M>{'j'}</M>-th entry of <M>{'(W^{(l+1)})^{\\top}\\delta^{(l+1)}'}</M>, which is the
            matrix form quoted above.
          </p>
          <p>
            <strong>3. The weight gradient.</strong> <M>{'W^{(l)}_{ji}'}</M> appears in exactly one
            place in the whole network: the term <M>{'W^{(l)}_{ji}a^{(l-1)}_i'}</M> inside{' '}
            <M>{'z^{(l)}_j'}</M>. So the chain-rule sum collapses to a single term:
          </p>
          <Equation plain>
            {'\\frac{\\partial L}{\\partial W^{(l)}_{ji}} = \\frac{\\partial L}{\\partial z^{(l)}_j}\\,\\frac{\\partial z^{(l)}_j}{\\partial W^{(l)}_{ji}} = \\delta^{(l)}_j\\,a^{(l-1)}_i'}
          </Equation>
          <p>
            <strong>4. The bias gradient.</strong> Identically,{' '}
            <M>{'\\partial z^{(l)}_j/\\partial b^{(l)}_j = 1'}</M>, so{' '}
            <M>{'\\partial L/\\partial b^{(l)}_j = \\delta^{(l)}_j'}</M>.
          </p>
          <p>
            Steps 3 and 4 are the reason the backward pass is cheap. Once{' '}
            <M>{'\\delta^{(l)}'}</M> is known, every parameter gradient in layer{' '}
            <M>{'l'}</M> is one multiplication — no further differentiation is needed. All the work
            is in the recursion of step 2, and that recursion is shared by every weight in the
            layer.
          </p>
        </Detail>

        <Detail kicker="derivation" title="Why softmax with cross-entropy collapses to ŷ − y">
          <p>
            With <M>{'L = -\\sum_k y_k \\log p_k'}</M> and{' '}
            <M>{'p = \\mathrm{softmax}(z)'}</M>, the two factors are{' '}
            <M>{'\\partial L/\\partial p_k = -y_k/p_k'}</M> and the softmax Jacobian{' '}
            <M>{'\\partial p_k/\\partial z_j = p_k(\\delta_{kj} - p_j)'}</M>. Multiply and sum:
          </p>
          <Equation plain>
            {'\\frac{\\partial L}{\\partial z_j} = \\sum_k \\left(-\\frac{y_k}{p_k}\\right)p_k(\\delta_{kj}-p_j) = -\\sum_k y_k(\\delta_{kj}-p_j)'}
          </Equation>
          <p>
            The <M>{'p_k'}</M> factors cancel exactly. Split the remaining sum:{' '}
            <M>{'\\sum_k y_k\\delta_{kj} = y_j'}</M> and{' '}
            <M>{'\\sum_k y_k p_j = p_j\\sum_k y_k = p_j'}</M> because the target sums to 1.
            Therefore
          </p>
          <Equation plain>{'\\frac{\\partial L}{\\partial z_j} = p_j - y_j'}</Equation>
          <p>
            The same cancellation happens for a sigmoid with binary cross-entropy:{' '}
            <M>{'\\partial L/\\partial \\hat{y} = (\\hat{y}-y)/(\\hat{y}(1-\\hat{y}))'}</M>{' '}
            multiplied by <M>{"\\sigma'(z) = \\hat{y}(1-\\hat{y})"}</M> leaves{' '}
            <M>{'\\hat{y} - y'}</M>.
          </p>
          <p>
            This is not a coincidence of algebra. It happens because cross-entropy is the negative
            log-likelihood of the distribution that the output activation parameterises — the pair is
            matched, and the matched pair always produces the difference between prediction and
            target. The practical consequences are a gradient that never saturates and an
            implementation that needs no division, which is why the two are fused into one operation
            in every framework.
          </p>
        </Detail>
      </div>

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
            <table className="data">
              <thead>
                <tr>
                  <th>factor</th>
                  <th>value</th>
                </tr>
              </thead>
              <tbody>
                {chainNumbers.map((item) => (
                  <tr key={item.label}>
                    <td>{item.label}</td>
                    <td>{fmt(item.value, 4)}</td>
                  </tr>
                ))}
                <tr className="is-current">
                  <td>product</td>
                  <td>{fmt(chainProduct, 6)}</td>
                </tr>
                <tr>
                  <td>backpropagation</td>
                  <td>{fmt(selectedGrad, 6)}</td>
                </tr>
                {numericCheck !== null ? (
                  <tr>
                    <td>
                      finite differences
                      <br />
                      <span className="faint">|difference|</span>
                    </td>
                    <td>
                      {fmt(numericCheck, 6)}
                      <br />
                      <span className="faint">{fmt(Math.abs(numericCheck - selectedGrad), 9)}</span>
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
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
          <h3 className="subhead">The algorithm</h3>
          <Steps>
            <li>
              Run the forward pass, storing every <M>{'\\mathbf{z}^{(l)}'}</M> and{' '}
              <M>{'\\mathbf{a}^{(l)}'}</M>.
            </li>
            <li>
              Form <M>{'\\delta^{(L)}'}</M> at the output from{' '}
              <M>{'\\partial L/\\partial \\hat{\\mathbf{y}}'}</M> and the output activation's
              derivative.
            </li>
            <li>
              For <M>{'l = L, L-1, \\ldots, 1'}</M>: record{' '}
              <M>{'\\partial L/\\partial W^{(l)} = \\delta^{(l)}(\\mathbf{a}^{(l-1)})^{\\top}'}</M>{' '}
              and <M>{'\\partial L/\\partial \\mathbf{b}^{(l)} = \\delta^{(l)}'}</M>, then
              propagate{' '}
              <M>{"\\delta^{(l-1)} = (W^{(l)})^{\\top}\\delta^{(l)} \\odot f'(\\mathbf{z}^{(l-1)})"}</M>.
            </li>
            <li>
              Hand the collected gradients to the optimiser. Backpropagation computes derivatives;
              it does not update anything.
            </li>
          </Steps>
          <p>
            That last point is worth separating. Backpropagation and gradient descent are often
            spoken of together but are independent: backpropagation answers "what is the
            derivative", the optimiser decides "what to do with it". Swapping SGD for Adam changes
            step 4 and nothing else.
          </p>

          <h3 className="subhead">Why work backwards</h3>
          <p>
            Both directions compute exact derivatives; they differ in what a single pass produces.
            Forward mode propagates <M>{'\\partial(\\cdot)/\\partial \\theta_k'}</M> for one
            chosen input <M>{'\\theta_k'}</M>, so one pass gives the derivative of{' '}
            <em>everything</em> with respect to <em>one</em> parameter. Reverse mode propagates{' '}
            <M>{'\\partial L/\\partial(\\cdot)'}</M> for one chosen output, so one pass gives
            the derivative of <em>one</em> scalar with respect to <em>everything</em>.
          </p>
          <p>
            Training needs the second shape: one scalar loss, millions of parameters. With{' '}
            <M>{'P'}</M> parameters, forward mode needs <M>{'P'}</M> passes and reverse mode needs
            one, so reverse mode is <M>{'P'}</M> times cheaper.
          </p>
          <Detail kicker="numbers" title="What that factor costs in practice">
            <p>
              Take a modest network with <M>{'P = 10^{6}'}</M> parameters and a forward pass that
              takes 1 ms.
            </p>
            <ul>
              <li>
                <strong>Reverse mode:</strong> one forward pass plus a backward pass costing roughly
                twice as much — about 3 ms per gradient.
              </li>
              <li>
                <strong>Forward mode:</strong> <M>{'10^{6}'}</M> passes at 1 ms each — about 17
                minutes per gradient.
              </li>
              <li>
                <strong>Finite differences:</strong> two evaluations per parameter, so{' '}
                <M>{'2\\times 10^{6}'}</M> passes — about 33 minutes per gradient, and the result
                is only approximate.
              </li>
            </ul>
            <p>
              A single training run needs tens of thousands of gradients. At 3 ms that is a couple
              of minutes; at 17 minutes it is several years. This factor is the reason neural
              networks are trained at all, and it is why the finite-difference check in the panel
              above is a test rather than a method.
            </p>
          </Detail>
        </div>
        <div className="prose-block">
          <h3 className="subhead">Where gradients vanish</h3>
          <p>
            Unrolling the recursion from the loss back to layer <M>{'l'}</M> gives a product of
            matrices:
          </p>
          <Equation plain>
            {"\\delta^{(l)} = \\left[\\prod_{k=l+1}^{L} D^{(k-1)}\\left(W^{(k)}\\right)^{\\top}\\right]\\delta^{(L)}, \\quad D^{(k)} = \\operatorname{diag}\\!\\left(f'(\\mathbf{z}^{(k)})\\right)"}
          </Equation>
          <p>
            Taking norms and using{' '}
            <M>{'\\lVert AB\\rVert \\le \\lVert A\\rVert\\lVert B\\rVert'}</M>:
          </p>
          <Equation plain>
            {"\\lVert\\delta^{(l)}\\rVert \\le \\left(\\max_z |f'(z)| \\cdot \\max_k \\lVert W^{(k)}\\rVert_2\\right)^{L-l}\\lVert\\delta^{(L)}\\rVert"}
          </Equation>
          <p>
            The bracket is a single number raised to the depth. Anything other than 1 compounds:
          </p>
          <table className="data" style={{ marginBottom: 14 }}>
            <thead>
              <tr>
                <th>per-layer factor</th>
                <th>after 10 layers</th>
                <th>after 30 layers</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>0.25 (sigmoid, ‖W‖ = 1)</td>
                <td>10⁻⁶</td>
                <td>10⁻¹⁸</td>
              </tr>
              <tr>
                <td>0.9</td>
                <td>0.35</td>
                <td>0.04</td>
              </tr>
              <tr>
                <td>1.0</td>
                <td>1</td>
                <td>1</td>
              </tr>
              <tr>
                <td>1.1</td>
                <td>2.6</td>
                <td>17</td>
              </tr>
              <tr>
                <td>1.5</td>
                <td>58</td>
                <td>1.9 × 10⁵</td>
              </tr>
            </tbody>
          </table>
          <p>
            A 32-bit float underflows to zero below about <M>{'10^{-38}'}</M>, so a sigmoid stack of
            around 60 layers produces gradients that are not merely small but exactly zero. The
            initialisation probe in section 03 measures this product directly.
          </p>
          <p>
            Every standard remedy targets the same bracket. ReLU raises{' '}
            <M>{"\\max|f'|"}</M> from 0.25 to 1. He initialisation sets{' '}
            <M>{'\\lVert W\\rVert'}</M> so the product starts near 1. Normalisation layers reset
            the scale at every layer rather than letting it drift. Residual connections add an
            identity path, replacing the product{' '}
            <M>{'\\prod D W^{\\top}'}</M> with <M>{'\\prod (I + DW^{\\top})'}</M>, which
            contains a term equal to 1 no matter how small the rest becomes — which is why they
            enable networks hundreds of layers deep.
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
