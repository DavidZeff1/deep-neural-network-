import type { ComponentType } from 'react';
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
  component: ComponentType<SectionProps>;
}

export const SECTIONS: SectionEntry[] = [
  { id: 'structure', title: 'Network structure', component: StructureSection },
  { id: 'neurons', title: 'Neurons', component: NeuronsSection },
  { id: 'weights', title: 'Weights & biases', component: WeightsSection },
  { id: 'activations', title: 'Activation functions', component: ActivationsSection },
  { id: 'forward', title: 'Forward propagation', component: ForwardSection },
  { id: 'loss', title: 'Loss functions', component: LossSection },
  { id: 'gradient-descent', title: 'Gradient descent', component: GradientDescentSection },
  { id: 'backprop', title: 'Backpropagation', component: BackpropSection },
  { id: 'training', title: 'Training', component: TrainingSection },
  { id: 'overfitting', title: 'Overfitting & regularisation', component: OverfittingSection },
  { id: 'depth', title: 'Why depth matters', component: DepthSection },
  { id: 'playground', title: 'Playground', component: PlaygroundSection },
];
