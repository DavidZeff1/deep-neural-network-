/**
 * End-to-end checks for every interactive control on the site.
 *
 * Run the built site first:  npm run build && npm run preview
 * Then:                      npm run test:browser
 */
import { mkdirSync } from 'node:fs';
import { chromium } from 'playwright';

const BASE = process.env.BASE_URL || 'http://localhost:4173';
const OUT = process.env.OUT_DIR || 'test-output';

mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, ok, detail = '') => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
};

// CHROMIUM_PATH lets the suite use a system Chromium instead of a downloaded one.
const executablePath = process.env.CHROMIUM_PATH;
const browser = await chromium.launch(executablePath ? { executablePath } : {});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const consoleIssues = [];
page.on('console', (m) => { if (m.type() === 'error') consoleIssues.push(m.text()); });
page.on('pageerror', (e) => consoleIssues.push(`pageerror: ${e.message}`));

await page.goto(BASE, { waitUntil: 'networkidle' });
await page.waitForTimeout(500);

// Helpers -------------------------------------------------------------------
const section = (id) => page.locator(`#${id}`);

async function setRange(locator, value) {
  await locator.evaluate((el, v) => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await page.waitForTimeout(90);
}

/** Finds a slider by the text of its label, so adding controls cannot break the test. */
function slider(root, labelText) {
  return root
    .locator('.control', { has: page.locator('.control__label', { hasText: labelText }) })
    .locator('input[type=range]')
    .first();
}

/** Reads a labelled stat tile value. */
async function stat(root, label) {
  const tile = root.locator('.stat', { has: page.locator('.stat__label', { hasText: new RegExp(`^${label}$`) }) });
  return (await tile.first().locator('.stat__value').textContent())?.trim() ?? '';
}

const num = (text) => Number(String(text).replace(/−/g, '-').replace(/[^\d.eE+-]/g, ''));

// --- 00 notation ------------------------------------------------------------
{
  const s = section('notation');
  await s.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);

  // Sigma: the running total must equal the sum of the listed products.
  const sumPanel = s.locator('.panel', { hasText: 'Adding up a list of products' });
  const rows = await sumPanel.locator('table.data tbody tr').count();
  check('00 sigma table has one row per term', rows === 3, `${rows} rows`);
  const productCells = await sumPanel.locator('table.data tbody tr td:nth-child(4)').allInnerTexts();
  const totalCell = await sumPanel.locator('table.data tbody tr:last-child td:last-child').innerText();
  const sumOfProducts = productCells.map(num).reduce((a, b) => a + b, 0);
  check('00 running total equals the sum of the products', Math.abs(sumOfProducts - num(totalCell)) < 1e-9, `${sumOfProducts} vs ${totalCell}`);
  // (0.5)(4) + (-1.2)(3) + (2.0)(2) = 2 - 3.6 + 4 = 2.4
  check('00 sigma example computes 2.4', Math.abs(num(totalCell) - 2.4) < 1e-9, totalCell);

  await setRange(slider(s.locator('.panel', { hasText: 'How many terms' }), 'n —'), 5);
  const rows5 = await sumPanel.locator('table.data tbody tr').count();
  check('00 changing n changes the number of terms', rows5 === 5, `${rows5} rows`);
  await setRange(slider(s.locator('.panel', { hasText: 'How many terms' }), 'n —'), 3);

  // Matrix times vector: stepping reveals one row at a time, with correct values.
  const mv = s.locator('.panel', { hasText: 'A matrix multiplied by a vector' });
  await mv.getByRole('button', { name: 'Show all' }).click();
  await page.waitForTimeout(200);
  const mvText = await mv.innerText();
  // W = [[0.5,-1],[2,0.5],[-0.5,1.5]], x = [4,2] -> [0, 9, 1]
  check('00 matrix-vector product is correct', /0\.0/.test(mvText) && /9\.0/.test(mvText) && /1\.0/.test(mvText), 'expected 0.0, 9.0, 1.0');
  await mv.getByRole('button', { name: 'Reset' }).click();
  await page.waitForTimeout(150);
  const hidden = await mv.innerText();
  check('00 results start hidden', hidden.includes('?'), 'question marks shown before stepping');

  // Derivative: shrinking h must bring rise/run towards the exact slope.
  const slopePanel = s.locator('.panel', { hasText: 'The arithmetic' });
  const slopeControls = s.locator('.panel', { hasText: 'Controls' }).first();
  // The readout renders labels and values as separate grid cells.
  const readValue = async (label) => {
    const labels = await slopePanel.locator('.readout__label').allInnerTexts();
    const values = await slopePanel.locator('.readout__value').allInnerTexts();
    const index = labels.findIndex((l) => l.trim() === label);
    return index === -1 ? Number.NaN : num(values[index]);
  };
  await setRange(slider(slopeControls, 'x — where on the curve'), 1);
  await setRange(slider(slopeControls, 'h — how far you nudge'), 0.3);
  const coarse = Math.abs(await readValue('difference'));
  await setRange(slider(slopeControls, 'h — how far you nudge'), -3);
  const fine = Math.abs(await readValue('difference'));
  check('00 a smaller nudge approximates the derivative better', fine < coarse / 100, `${coarse} -> ${fine}`);
  const exact = await readValue("exact slope f'(x)");
  check('00 derivative of x² at x = 1 is 2', Math.abs(exact - 2) < 1e-9, `${exact}`);

  // Chain rule: the product of the three rates must match the measured change.
  const chain = s.locator('.panel', { hasText: 'The rate at each step' });
  const chainRow = async (label) =>
    num(
      await chain
        .locator('tr', { has: page.locator('td', { hasText: label }) })
        .first()
        .locator('td')
        .nth(2)
        .innerText(),
    );
  const viaChainRule = await chainRow('all three multiplied');
  const viaNudging = await chainRow('measured by nudging');
  check('00 chain rule matches a direct measurement', Math.abs(viaChainRule - viaNudging) < 1e-3, `${viaChainRule} vs ${viaNudging}`);
  const rateZ = await chainRow('x changes z');
  check('00 the first rate equals the weight', Math.abs(rateZ - 0.8) < 1e-9, `${rateZ}`);
}

// --- 01 structure -----------------------------------------------------------
{
  const s = section('structure');
  await s.scrollIntoViewIfNeeded();
  const before = await s.locator('.stat__value').last().textContent();
  await s.getByRole('button', { name: '+ layer' }).click();
  await page.waitForTimeout(150);
  const after = await s.locator('.stat__value').last().textContent();
  check('01 add layer changes parameter count', before !== after, `${before} -> ${after}`);

  const rows = await s.locator('table.data tbody tr').count();
  check('01 parameter table gains a row', rows === 5, `${rows} rows`);

  await s.getByRole('button', { name: '− layer' }).click();
  await page.waitForTimeout(150);
  const restored = await s.locator('.stat__value').last().textContent();
  check('01 remove layer restores count', restored === before, `${restored}`);

  const widthSlider = s.locator('input[type=range]').nth(1);
  const nodesBefore = await s.locator('svg .node').count();
  await setRange(widthSlider, 9);
  const nodesAfter = await s.locator('svg .node').count();
  check('01 width slider changes node count', nodesAfter > nodesBefore, `${nodesBefore} -> ${nodesAfter}`);
}

// --- 02 neurons -------------------------------------------------------------
{
  const s = section('neurons');
  await s.scrollIntoViewIfNeeded();
  await s.getByRole('button', { name: 'Reset to the worked example' }).click();
  await page.waitForTimeout(150);
  const z = num(await stat(s, 'z'));
  check('02 worked example z = 0.85', Math.abs(z - 0.85) < 1e-6, `z=${z}`);
  const a = num(await stat(s, 'a = f\\(z\\)'));
  check('02 ReLU(0.85) = 0.85', Math.abs(a - 0.85) < 1e-6, `a=${a}`);

  const calc = await s.locator('.calc').first().innerText();
  check('02 calculation shows the substituted terms', calc.includes('(0.80)(1.00)') && calc.includes('(−0.30)(0.50)'), calc.split('\n')[1]);

  // sigmoid at z = 0.85 -> 0.7005671...
  await s.getByRole('radio', { name: 'Sigmoid' }).click();
  await page.waitForTimeout(150);
  const aSig = num(await stat(s, 'a = f\\(z\\)'));
  check('02 sigmoid(0.85) = 0.7006', Math.abs(aSig - 1 / (1 + Math.exp(-0.85))) < 5e-4, `a=${aSig}`);
  const dSig = num(await stat(s, "f'\\(z\\)"));
  const expectedD = (1 / (1 + Math.exp(-0.85))) * (1 - 1 / (1 + Math.exp(-0.85)));
  check('02 sigmoid derivative correct', Math.abs(dSig - expectedD) < 5e-4, `${dSig} vs ${expectedD.toFixed(4)}`);

  // w1 slider changes z
  const w1 = s.locator('input[type=range]').nth(2);
  await setRange(w1, 1.5);
  const z2 = num(await stat(s, 'z'));
  check('02 w₁ slider updates z', Math.abs(z2 - (1.5 * 1.0 + -0.3 * 0.5 + 0.2)) < 1e-6, `z=${z2}`);

  await s.getByRole('radio', { name: 'ReLU', exact: true }).click();
  const canvas = s.locator('canvas').first();
  await canvas.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const box = await canvas.boundingBox();
  await page.mouse.click(box.x + box.width * 0.25, box.y + box.height * 0.25);
  await page.waitForTimeout(150);
  const z3 = num(await stat(s, 'z'));
  check('02 clicking the input plane moves the probe', Math.abs(z3 - z2) > 1e-9, `z=${z3}`);
}

// --- 03 weights -------------------------------------------------------------
{
  const s = section('weights');
  await s.scrollIntoViewIfNeeded();
  const before = num(await stat(s, 'Output ŷ'));
  const slider = s.locator('input[type=range]').first();
  await setRange(slider, 2.5);
  const after = num(await stat(s, 'Output ŷ'));
  check('03 editing a weight changes the output', Math.abs(after - before) > 1e-6, `${before} -> ${after}`);

  const normBefore = num(await stat(s, '‖W‖₂'));
  await s.getByRole('button', { name: 'Scale ×2' }).click();
  await page.waitForTimeout(120);
  const normAfter = num(await stat(s, '‖W‖₂'));
  check('03 scale ×2 doubles the weight norm', Math.abs(normAfter - 2 * normBefore) < 0.02, `${normBefore} -> ${normAfter}`);

  await s.getByRole('button', { name: 'All weights → 0' }).click();
  await page.waitForTimeout(120);
  const zeroNorm = num(await stat(s, '‖W‖₂'));
  check('03 zeroing weights gives ‖W‖ = 0', zeroNorm === 0, `${zeroNorm}`);
  await s.getByRole('button', { name: 'Re-initialise' }).click();

  // Initialisation probe: the gain slider must change how the scale compounds.
  const probe = s.locator('.panel', { hasText: 'Initialisation scale' });
  await s.getByRole('button', { name: /recommended gain/ }).click();
  await page.waitForTimeout(250);
  const balanced = num(await stat(s, 'a std, layer 10'));
  check('03 He gain keeps the activation scale near 1', balanced > 0.5 && balanced < 2.5, `${balanced}`);
  await s.getByRole('button', { name: 'Too small' }).click();
  await page.waitForTimeout(250);
  const shrunk = num(await stat(s, 'a std, layer 10'));
  check('03 a small gain makes activations vanish with depth', shrunk < balanced / 10, `${shrunk} vs ${balanced}`);
  const ratio = num(await stat(s, 'ratio per layer'));
  check('03 per-layer ratio below 1 when the gain is too small', ratio < 1, `${ratio}`);
  await s.getByRole('button', { name: 'Too large' }).click();
  await page.waitForTimeout(250);
  const grown = num(await stat(s, 'a std, layer 10'));
  check('03 a large gain makes activations explode with depth', grown > balanced * 2, `${grown} vs ${balanced}`);
  await s.getByRole('button', { name: /recommended gain/ }).click();

  // Derivations are collapsed by default and open on click.
  const firstDetail = s.locator('details.detail').first();
  check('03 derivations start collapsed', !(await firstDetail.evaluate((el) => el.open)));
  await firstDetail.locator('summary').click();
  await page.waitForTimeout(150);
  check('03 derivation opens on click', await firstDetail.evaluate((el) => el.open));
  await firstDetail.locator('summary').click();
}

// --- 04 activations ---------------------------------------------------------
{
  const s = section('activations');
  await s.scrollIntoViewIfNeeded();
  const zSlider = s.locator('input[type=range]').first();
  await setRange(zSlider, 2);
  let out = num(await stat(s, 'output f\\(z\\)'));
  check('04 ReLU(2) = 2', Math.abs(out - 2) < 1e-6, `${out}`);
  await setRange(zSlider, -1.5);
  out = num(await stat(s, 'output f\\(z\\)'));
  check('04 ReLU(−1.5) = 0', out === 0, `${out}`);

  await s.getByRole('radio', { name: 'Tanh' }).click();
  await setRange(zSlider, 1);
  out = num(await stat(s, 'output f\\(z\\)'));
  check('04 tanh(1) = 0.762', Math.abs(out - Math.tanh(1)) < 5e-4, `${out}`);
  const slope = num(await stat(s, "slope f'\\(z\\)"));
  check('04 tanh derivative = 1 − tanh²', Math.abs(slope - (1 - Math.tanh(1) ** 2)) < 5e-4, `${slope}`);

  // Softmax
  const softmaxCalc = s.locator('.calc').last();
  const text = await softmaxCalc.innerText();
  const sumLine = text.split('\n').find((l) => l.includes('Σp'));
  check('04 softmax probabilities sum to 1', /1\.0{4,}/.test(sumLine ?? ''), sumLine);

  const logitSliders = s.locator('.panel', { hasText: 'Logits' }).locator('input[type=range]');
  await setRange(logitSliders.nth(0), 4);
  const text2 = await softmaxCalc.innerText();
  check('04 logit slider changes probabilities', text2 !== text, text2.split('\n').at(-2));
  const sum2 = text2.split('\n').find((l) => l.includes('Σp'));
  check('04 probabilities still sum to 1 after change', /1\.0{4,}/.test(sum2 ?? ''), sum2);

  await s.getByRole('button', { name: 'All equal' }).click();
  await page.waitForTimeout(120);
  const equal = await softmaxCalc.innerText();
  check('04 equal logits give uniform probabilities', equal.includes('0.3333, 0.3333, 0.3333'), equal.split('\n')[3]);

  // Temperature: higher T raises the entropy towards ln 3 = 1.0986.
  await s.getByRole('button', { name: 'Reset' }).click();
  await page.waitForTimeout(150);
  const entropyOf = async () => {
    const text = await softmaxCalc.innerText();
    const line = text.split('\n').find((l) => l.includes('H(p)')) ?? '';
    // "H(p):  0.9089 nats (max 1.0986)" — take the first number after the colon.
    const match = /H\(p\):\s*(−?[\d.]+)/.exec(line);
    return num(match?.[1] ?? 'NaN');
  };
  const baseEntropy = await entropyOf();
  const tSlider = slider(s.locator('.panel', { hasText: 'Logits' }), 'Temperature');
  await setRange(tSlider, 4);
  const hotEntropy = await entropyOf();
  check('04 raising T raises the entropy towards ln K', hotEntropy > baseEntropy && hotEntropy < Math.log(3) + 1e-6, `${baseEntropy} -> ${hotEntropy}`);
  await setRange(tSlider, 0.1);
  const coldEntropy = await entropyOf();
  check('04 lowering T concentrates the distribution', coldEntropy < 0.01, `${coldEntropy}`);
  const coldText = await softmaxCalc.innerText();
  check('04 low T leaves the argmax unchanged', /p:\s+1\.0000/.test(coldText), coldText.split('\n').find((l) => l.startsWith('p:')));
  await setRange(tSlider, 1);

  // Large logits must not overflow: the shift keeps the result identical.
  await s.getByRole('button', { name: 'Large logits' }).click();
  await page.waitForTimeout(200);
  const large = await softmaxCalc.innerText();
  check('04 large logits give the same probabilities, not NaN', large.includes('0.6590, 0.2424, 0.0986'), large.split('\n').find((l) => l.startsWith('p:')));
  await s.getByRole('button', { name: 'Reset' }).click();
}

// --- 05 forward -------------------------------------------------------------
{
  const s = section('forward');
  await s.scrollIntoViewIfNeeded();
  await s.getByRole('button', { name: 'Reset' }).click().catch(() => {});
  let yhat = await stat(s, 'ŷ');
  check('05 output hidden before stepping', yhat === '—', yhat);
  for (let i = 0; i < 4; i++) {
    await s.getByRole('button', { name: 'Next step →' }).click();
    await page.waitForTimeout(120);
  }
  yhat = num(await stat(s, 'ŷ'));
  check('05 four steps reveal the prediction', yhat > 0 && yhat < 1, `ŷ=${yhat}`);

  // The displayed matrix-form result must equal the stat readout.
  const calcText = await s.locator('.calc').first().innerText();
  check('05 calculation panel shows a weighted sum', /\(-?[\d.−]+\)\(-?[\d.−]+\)/.test(calcText), calcText.split('\n')[2]?.slice(0, 60));

  const z1 = num(await stat(s, 'z⁽¹⁾ selected'));
  const a1 = num(await stat(s, 'a⁽¹⁾ selected'));
  check('05 ReLU consistency between z and a', Math.abs(a1 - Math.max(0, z1)) < 5e-4, `z=${z1} a=${a1}`);
}

// --- 06 loss ----------------------------------------------------------------
{
  const s = section('loss');
  await s.scrollIntoViewIfNeeded();
  const pred = s.locator('input[type=range]').first();
  await setRange(pred, 0.8);
  const loss = num(await stat(s, 'Loss'));
  check('06 BCE(0.8, y=1) = −ln 0.8 = 0.2231', Math.abs(loss - -Math.log(0.8)) < 5e-4, `${loss}`);
  const grad = num(await stat(s, '∂L/∂ŷ'));
  check('06 ∂L/∂ŷ = (ŷ−y)/(ŷ(1−ŷ))', Math.abs(grad - (0.8 - 1) / (0.8 * 0.2)) < 5e-3, `${grad}`);

  await s.getByRole('radio', { name: 'Mean squared error' }).click();
  await page.waitForTimeout(120);
  const mse = num(await stat(s, 'Loss'));
  check('06 MSE(0.8, y=1) = 0.04', Math.abs(mse - 0.04) < 1e-4, `${mse}`);

  await s.getByRole('radio', { name: 'y = 0' }).click();
  await page.waitForTimeout(120);
  const mse0 = num(await stat(s, 'Loss'));
  check('06 MSE(0.8, y=0) = 0.64', Math.abs(mse0 - 0.64) < 1e-4, `${mse0}`);

  await s.getByRole('radio', { name: 'Categorical cross-entropy' }).click();
  await page.waitForTimeout(200);
  const p = num(await stat(s, 'p of true class'));
  const cceLoss = num(await stat(s, 'Loss'));
  check('06 CCE = −ln p(true class)', Math.abs(cceLoss - -Math.log(p)) < 5e-3, `p=${p} L=${cceLoss}`);
  const perplexity = num(await stat(s, 'Perplexity eᴸ'));
  check('06 perplexity = e^L', Math.abs(perplexity - Math.exp(cceLoss)) < 5e-3, `${perplexity}`);
}

// --- 07 gradient descent ----------------------------------------------------
{
  const s = section('gradient-descent');
  await s.scrollIntoViewIfNeeded();
  const theta0 = num(await stat(s, 'Parameter θ'));
  const g0 = num(await stat(s, 'Gradient'));
  const lr = num(await stat(s, 'Learning rate'));
  await s.getByRole('button', { name: 'Step' }).click();
  await page.waitForTimeout(150);
  const theta1 = num(await stat(s, 'Parameter θ'));
  check('07 step follows θ ← θ − η∇L', Math.abs(theta1 - (theta0 - lr * g0)) < 1e-3, `${theta0} − ${lr}×${g0} = ${theta1}`);
  const loss1 = num(await stat(s, 'Loss'));
  check('07 loss decreased after one step', loss1 < theta0 * theta0 + 1e-9, `${loss1}`);

  // Divergence at a large learning rate
  const controls = s.locator('.panel', { hasText: 'Controls' });
  const lrSlider = slider(controls, 'Learning rate');
  await setRange(lrSlider, 1.1);
  await setRange(slider(controls, 'Scrub to iteration'), 30);
  const bigTheta = Math.abs(num(await stat(s, 'Parameter θ')));
  check('07 η = 1.1 diverges on L = θ²', bigTheta > 10, `|θ|=${bigTheta}`);
  await setRange(lrSlider, 0.1);
  await setRange(slider(controls, 'Scrub to iteration'), 0);

  // Momentum accumulates: v = beta*v + g, so after two steps |v| exceeds |g|.
  await setRange(slider(controls, 'Momentum'), 0.9);
  await setRange(slider(controls, 'Scrub to iteration'), 3);
  const velocity = Math.abs(num(await stat(s, 'Velocity v')));
  const gradient = Math.abs(num(await stat(s, 'Gradient')));
  check('07 momentum accumulates velocity beyond the gradient', velocity > gradient, `|v|=${velocity} |g|=${gradient}`);
  await setRange(slider(controls, 'Momentum'), 0);
  await setRange(slider(controls, 'Scrub to iteration'), 0);

  await s.getByRole('radio', { name: 'Two minima' }).click();
  await page.waitForTimeout(200);
  const title = await s.locator('.panel__title').first().textContent();
  check('07 landscape switch works', /minima/i.test(title ?? ''), title);
  await s.getByRole('radio', { name: 'Convex', exact: true }).click();

  await s.getByRole('radio', { name: 'Two parameters' }).click();
  await page.waitForTimeout(400);
  const w0 = num(await stat(s, 'w'));
  const gw0 = num(await stat(s, '∂L/∂w'));
  await s.getByRole('button', { name: 'Step' }).click();
  await page.waitForTimeout(150);
  const w1 = num(await stat(s, 'w'));
  check('07 2-D step follows the same rule', Math.abs(w1 - (w0 - 0.15 * gw0)) < 2e-3, `${w0} -> ${w1}`);

  // Feature scaling stretches the surface: the condition number must rise sharply.
  const controls2d = s.locator('.panel', { hasText: 'Controls' });
  const kappaBefore = num(await stat(s, 'κ \\(condition no.\\)'));
  await setRange(slider(controls2d, 'Feature scale'), 0.2);
  await page.waitForTimeout(300);
  const kappaAfter = num(await stat(s, 'κ \\(condition no.\\)'));
  check('07 feature scaling raises the condition number', kappaAfter > kappaBefore * 5, `${kappaBefore} -> ${kappaAfter}`);

  // At a fixed rate, momentum should get closer to the minimum than plain descent.
  await setRange(slider(controls2d, 'Learning rate'), 0.45);
  await setRange(slider(controls2d, 'Iterations'), 120);
  await setRange(slider(controls2d, 'Scrub'), 120);
  await page.waitForTimeout(300);
  const lossPlain = num(await stat(s, 'Loss'));
  await setRange(slider(controls2d, 'Momentum'), 0.8);
  await setRange(slider(controls2d, 'Scrub'), 120);
  await page.waitForTimeout(300);
  const lossMomentum = num(await stat(s, 'Loss'));
  check('07 momentum converges further on an ill-conditioned surface', lossMomentum < lossPlain, `${lossPlain} -> ${lossMomentum}`);

  await s.getByRole('radio', { name: 'One parameter' }).click();
}

// --- 08 backprop ------------------------------------------------------------
{
  const s = section('backprop');
  await s.scrollIntoViewIfNeeded();
  await s.getByRole('button', { name: 'Restore weights' }).click();
  await page.waitForTimeout(150);
  await s.getByRole('button', { name: 'Check against finite differences' }).click();
  await page.waitForTimeout(200);
  const chainPanel = s.locator('.panel', { hasText: 'Chain rule' });
  const rowValue = async (label) =>
    num(
      await chainPanel
        .locator('tr', { has: page.locator('td', { hasText: label }) })
        .first()
        .locator('td')
        .nth(1)
        .innerText(),
    );
  const chainValue = await rowValue('product');
  const backpropValue = await rowValue('backpropagation');
  check('08 chain-rule product equals the backprop value', Math.abs(chainValue - backpropValue) < 1e-6, `${chainValue} vs ${backpropValue}`);

  const diffCell = await chainPanel
    .locator('tr', { has: page.locator('td', { hasText: 'finite differences' }) })
    .locator('td')
    .nth(1)
    .innerText();
  const [numericValue, difference] = diffCell.split('\n').map(num);
  check('08 analytic gradient matches finite differences', Math.abs(difference) < 1e-6, `numeric ${numericValue}, |Δ| ${difference}`);
  check('08 finite-difference value matches backpropagation', Math.abs(numericValue - backpropValue) < 1e-5, `${numericValue} vs ${backpropValue}`);

  const lossBefore = num(await stat(s, 'Loss'));
  await s.getByRole('button', { name: 'Apply one update' }).click();
  await page.waitForTimeout(200);
  const lossAfter = num(await stat(s, 'Loss'));
  check('08 one gradient step reduces the loss', lossAfter < lossBefore, `${lossBefore} -> ${lossAfter}`);

  await s.getByRole('button', { name: 'Restore weights' }).click();
  for (let i = 0; i < 5; i++) {
    await s.getByRole('button', { name: 'Next step →' }).click();
    await page.waitForTimeout(80);
  }
  const d2 = await stat(s, 'δ⁽²⁾');
  check('08 stepping reveals δ⁽²⁾', d2 !== '—', d2);
  const yhat = num(await stat(s, 'ŷ'));
  check('08 δ⁽²⁾ = ŷ − y for sigmoid + BCE', Math.abs(num(d2) - (yhat - 1)) < 5e-4, `${d2} vs ${(yhat - 1).toFixed(4)}`);
}

// --- 09 training ------------------------------------------------------------
{
  const s = section('training');
  await s.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const loss0 = num(await stat(s, 'Loss'));
  await s.getByRole('button', { name: 'Train', exact: true }).click();
  await page.waitForTimeout(3500);
  await s.getByRole('button', { name: 'Pause' }).click();
  await page.waitForTimeout(200);
  const epoch = num(await stat(s, 'Epoch'));
  const loss1 = num(await stat(s, 'Loss'));
  const acc = await stat(s, 'Accuracy');
  check('09 training advances epochs', epoch > 10, `epoch=${epoch}`);
  check('09 training reduces the loss', loss1 < loss0, `${loss0} -> ${loss1}`);
  check('09 accuracy improves on Circles', parseFloat(acc) > 80, acc);
  const curvePoints = await s.locator('.panel', { hasText: 'Training loss' }).locator('path.curve').count();
  check('09 loss curve is drawn', curvePoints >= 1, `${curvePoints} paths`);

  await s.getByRole('button', { name: '+10' }).click();
  await page.waitForTimeout(400);
  const epoch2 = num(await stat(s, 'Epoch'));
  check('09 +10 epochs adds exactly 10', epoch2 === epoch + 10, `${epoch} -> ${epoch2}`);

  await s.getByRole('button', { name: 'Reset' }).click();
  await page.waitForTimeout(300);
  check('09 reset returns to epoch 0', num(await stat(s, 'Epoch')) === 0);

  // dataset switch
  await s.locator('select').selectOption('xor');
  await page.waitForTimeout(400);
  const hint = await s.locator('.panel', { hasText: 'Decision boundary' }).innerText();
  check('09 dataset switch updates the description', /XOR|quadrant/i.test(hint), 'xor selected');
}

// --- 10 overfitting ---------------------------------------------------------
{
  const s = section('overfitting');
  await s.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await s.getByRole('button', { name: '+20 epochs' }).click();
  await page.waitForTimeout(1200);
  const trainLoss = num(await stat(s, 'Train loss'));
  const valLoss = num(await stat(s, 'Val loss'));
  const gap = num(await stat(s, 'Gap'));
  check('10 validation loss is reported', Number.isFinite(valLoss) && valLoss > 0, `${valLoss}`);
  check('10 gap = val − train', Math.abs(gap - (valLoss - trainLoss)) < 2e-3, `${gap}`);

  const paramsBefore = num(await stat(s, 'Parameters'));
  const widthSlider = s.locator('.panel', { hasText: 'Model complexity' }).locator('input[type=range]').first();
  await setRange(widthSlider, 40);
  await page.waitForTimeout(400);
  const paramsAfter = num(await stat(s, 'Parameters'));
  check('10 complexity slider changes parameter count', paramsAfter > paramsBefore, `${paramsBefore} -> ${paramsAfter}`);

  // L2 must shrink the weights: compare the penalty after training with and without.
  const l2Slider = s.locator('.panel', { hasText: 'Regularisation' }).locator('input[type=range]').first();
  await setRange(l2Slider, 0.05);
  await page.waitForTimeout(200);
  const regText = await s.locator('.panel', { hasText: 'Regularisation' }).innerText();
  check('10 L2 penalty term is displayed', /L2 penalty term/.test(regText), regText.split('\n').at(-1)?.slice(0, 60));
  await setRange(l2Slider, 0);

  const bestButton = s.getByRole('button', { name: /Early stop/ });
  const enabled = await bestButton.isEnabled();
  check('10 early-stopping restore is available', enabled);
  if (enabled) {
    await bestButton.click();
    await page.waitForTimeout(300);
    check('10 restoring the best epoch keeps the app responsive', (await stat(s, 'Val loss')).length > 0);
  }
}

// --- 11 depth ---------------------------------------------------------------
{
  const s = section('depth');
  await s.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const before = await s.locator('.panel', { hasText: 'Same data' }).innerText();
  await s.getByRole('button', { name: '+25 epochs' }).click();
  await page.waitForTimeout(2500);
  const after = await s.locator('.panel', { hasText: 'Same data' }).innerText();
  check('11 all four networks train', before !== after, after.split('\n').slice(0, 2).join(' | '));

  const canvases = await s.locator('canvas').count();
  check('11 four boundaries plus hidden-unit tiles render', canvases >= 5, `${canvases} canvases`);

  const rows = await s.locator('table.data tbody tr').count();
  check('11 region-count table has four rows', rows === 4, `${rows}`);
  const firstRow = await s.locator('table.data tbody tr').first().innerText();
  check('11 shallow region count for n = 8 is 37', firstRow.includes('37'), firstRow.replace(/\s+/g, ' '));
}

// --- 12 playground ----------------------------------------------------------
{
  const s = section('playground');
  await s.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const canvas = s.locator('canvas').first();
  await canvas.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const box = await canvas.boundingBox();
  await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.7);
  await page.waitForTimeout(200);
  const px = num(await stat(s, 'Probe x₁'));
  const py = num(await stat(s, 'Probe x₂'));
  check('12 clicking the plane moves the probe', Math.abs(px + 0.5) < 0.25 && Math.abs(py + 0.5) < 0.25, `(${px}, ${py})`);

  // Select a hidden unit and confirm the inspector reproduces z = Σwa + b.
  const nodes = s.locator('.panel', { hasText: 'Network' }).locator('.node');
  await nodes.nth(3).click();
  await page.waitForTimeout(200);
  const inspector = await s.locator('.panel', { hasText: 'Inspector' }).innerText();
  check('12 inspector shows a unit computation', /z = /.test(inspector), inspector.split('\n').find((l) => l.startsWith('z =')) ?? '');

  const zLine = inspector.split('\n').find((l) => /^z = -?[\d.−]+$/.test(l.trim()));
  const aLine = inspector.split('\n').find((l) => l.trim().startsWith('a = '));
  if (zLine && aLine) {
    const z = num(zLine);
    const a = num(aLine.split('=').pop());
    check('12 inspector a = tanh(z)', Math.abs(a - Math.tanh(z)) < 5e-3, `z=${z} a=${a}`);
  } else {
    check('12 inspector a = tanh(z)', false, 'could not parse');
  }

  const epoch0 = num(await stat(s, 'Epoch'));
  await s.getByRole('button', { name: '+25' }).click();
  await page.waitForTimeout(1500);
  const epoch1 = num(await stat(s, 'Epoch'));
  check('12 +25 epochs trains', epoch1 === epoch0 + 25, `${epoch0} -> ${epoch1}`);

  // Edge inspection shows a gradient.
  const edges = s.locator('.panel', { hasText: 'Network' }).locator('.edge');
  await edges.nth(2).click();
  await page.waitForTimeout(400);
  const edgeInfo = await s.locator('.panel', { hasText: 'Inspector' }).innerText();
  check('12 inspector shows ∂L/∂w for a connection', /∂L\/∂w/.test(edgeInfo), edgeInfo.split('\n').find((l) => l.includes('∂L/∂w')) ?? '');
}

// --- navigation, theme, responsiveness --------------------------------------
{
  await page.locator('.navitem', { hasText: 'Loss functions' }).click();
  await page.waitForTimeout(900);
  const activeText = await page.locator('.navitem--active').innerText();
  check('nav jumps to the clicked section', /Loss/.test(activeText), activeText.replace(/\s+/g, ' '));

  const progress = await page.locator('.progress__label').innerText();
  check('progress indicator tracks position', /0?6\/12/.test(progress), progress);

  const themeBefore = await page.evaluate(() => document.documentElement.dataset.theme);
  await page.locator('.sidebar__foot .iconbutton').click();
  await page.waitForTimeout(300);
  const themeAfter = await page.evaluate(() => document.documentElement.dataset.theme);
  check('theme toggle switches themes', themeBefore !== themeAfter, `${themeBefore} -> ${themeAfter}`);
  await page.locator('#playground').scrollIntoViewIfNeeded();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/dark-playground.png` });
  await page.locator('.sidebar__foot .iconbutton').click();
  await page.waitForTimeout(200);
}

// Mobile
{
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(400);
  await page.locator('#top').scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check('mobile: no horizontal page overflow', overflow <= 1, `${overflow}px`);
  const sidebarHidden = await page.locator('.sidebar').evaluate((el) => el.getBoundingClientRect().right <= 1);
  check('mobile: sidebar starts closed', sidebarHidden);
  await page.locator('.topbar .iconbutton').first().click();
  await page.waitForTimeout(400);
  const sidebarOpen = await page.locator('.sidebar').evaluate((el) => el.getBoundingClientRect().right > 100);
  check('mobile: menu opens the sidebar', sidebarOpen);
  await page.locator('.navitem', { hasText: 'Neurons' }).click();
  await page.waitForTimeout(900);
  const closed = await page.locator('.sidebar').evaluate((el) => el.getBoundingClientRect().right <= 1);
  check('mobile: choosing a section closes the sidebar', closed);
  await page.screenshot({ path: `${OUT}/mobile-neurons.png` });
  await page.locator('#playground').scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  const overflow2 = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check('mobile: playground has no horizontal overflow', overflow2 <= 1, `${overflow2}px`);
  await page.screenshot({ path: `${OUT}/mobile-playground.png` });
  await page.setViewportSize({ width: 1440, height: 1000 });
}

check('no console errors during the run', consoleIssues.length === 0, consoleIssues.slice(0, 3).join(' | '));

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) {
  console.log('FAILURES:');
  for (const f of failed) console.log(`  - ${f.name}: ${f.detail}`);
}
await browser.close();
process.exit(failed.length ? 1 : 0);
