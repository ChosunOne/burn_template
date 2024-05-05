use burn::{
    nn::{loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Linear, Relu},
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::Batch;

#[derive(Module, Debug)]
pub struct ClassificationModel<B: Backend> {
    activation: Relu,
    dropout: Dropout,
    linear: Linear<B>,
}

impl<B: Backend> ClassificationModel<B> {
    pub fn forward(&self, inputs: Tensor<B, 1>) -> Tensor<B, 2> {
        todo!()
    }

    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 1>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(inputs);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<Batch<B>, ClassificationOutput<B>> for ClassificationModel<B> {
    fn step(&self, batch: Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Batch<B>, ClassificationOutput<B>> for ClassificationModel<B> {
    fn step(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct ClassificationModelConfig {
    #[config(default = "0.5")]
    dropout: f64,
}

impl ClassificationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClassificationModel<B> {
        ClassificationModel {
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            ..todo!()
        }
    }
}
