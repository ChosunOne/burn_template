use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

use crate::{
    data::regression::{DatasetBatcher, DatasetItem},
    train::regression::TrainingConfig,
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: DatasetItem) -> f64 {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Failed to load model config");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Failed to load model");

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = DatasetBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.inputs);
    let prediction = output.flatten::<1>(0, 1).into_scalar();
    prediction.elem::<f64>()
}
