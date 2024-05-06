use burn::{
    nn::{
        loss::{MseLoss, Reduction},
        Dropout, DropoutConfig, Linear, Relu,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::regression::Batch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    activation: Relu,
    dropout: Dropout,
    linear: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, inputs: Tensor<B, 1>) -> Tensor<B, 2> {
        todo!()
    }

    pub fn forward_regression(
        &self,
        inputs: Tensor<B, 1>,
        targets: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let output = self.forward(inputs);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Sum);
        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<Batch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: Batch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Batch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: Batch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.inputs, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            ..todo!()
        }
    }
}
