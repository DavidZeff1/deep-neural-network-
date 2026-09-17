import type { ComponentType } from 'react';
import { NotationSection } from './00-notation.tsx';
import { StructureSection } from './01-structure.tsx';
import { NeuronsSection } from './02-neurons.tsx';
import { WeightsSection } from './03-weights.tsx';
import { ActivationsSection } from './04-activations.tsx';
import { ForwardSection } from './05-forward.tsx';
import { LossSection } from './06-loss.tsx';
import { GradientDescentSection } from './07-gradient-descent.tsx';
import { BackpropSection } from './08-backprop.tsx';
import { TrainingSection } from './09-training.tsx';
import { OverfittingSection } from './10-overfitting.tsx';
import { DepthSection } from './11-depth.tsx';
import { PlaygroundSection } from './12-playground.tsx';

export interface SectionProps {
  id: string;
  index: number;
}

export interface SectionEntry {
  id: string;
  title: string;
  /** Displayed in the navigation; the prerequisites section is 00. */
  number: string;
  component: ComponentType<SectionProps>;
}

export const SECTIONS: SectionEntry[] = [
  { id: 'notation', title: 'Reading the mathematics', number: '00', component: NotationSection },
  { id: 'structure', title: 'Network structure', number: '01', component: StructureSection },
  { id: 'neurons', number: '02', title: 'Neurons', component: NeuronsSection },
  { id: 'weights', number: '03', title: 'Weights & biases', component: WeightsSection },
  { id: 'activations', number: '04', title: 'Activation functions', component: ActivationsSection },
  { id: 'forward', number: '05', title: 'Forward propagation', component: ForwardSection },
  { id: 'loss', number: '06', title: 'Loss functions', component: LossSection },
  { id: 'gradient-descent', number: '07', title: 'Gradient descent', component: GradientDescentSection },
  { id: 'backprop', number: '08', title: 'Backpropagation', component: BackpropSection },
  { id: 'training', number: '09', title: 'Training', component: TrainingSection },
  { id: 'overfitting', number: '10', title: 'Overfitting & regularisation', component: OverfittingSection },
  { id: 'depth', number: '11', title: 'Why depth matters', component: DepthSection },
  { id: 'playground', number: '12', title: 'Playground', component: PlaygroundSection },
];
