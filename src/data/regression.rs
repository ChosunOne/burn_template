use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DatasetItem {
    pub value: f64,
    pub inputs: Vec<f64>,
}

#[derive(Clone)]
pub struct DatasetBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DatasetBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 1>,
    pub targets: Tensor<B, 2, Float>,
}

impl<B: Backend> Batcher<DatasetItem, Batch<B>> for DatasetBatcher<B> {
    fn batch(&self, items: Vec<DatasetItem>) -> Batch<B> {
        todo!()
    }
}

pub struct Data {}

impl Data {
    pub fn train() -> Self {
        todo!()
    }

    pub fn valid() -> Self {
        todo!()
    }

    pub fn test() -> Self {
        todo!()
    }
}

impl Dataset<DatasetItem> for Data {
    fn get(&self, index: usize) -> Option<DatasetItem> {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }
}
