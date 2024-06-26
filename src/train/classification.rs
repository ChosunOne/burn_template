use std::path::Path;

use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, CudaMetric, LearningRateMetric,
            LossMetric,
        },
        LearnerBuilder,
    },
};

use crate::{
    data::classification::{Data, DatasetBatcher},
    model::classification::ModelConfig,
};

use super::{create_artifact_dir, get_last_epoch};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    #[config(default = 0)]
    pub starting_epoch: usize,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    fresh: bool,
) {
    let artifact_dir_path = Path::new(artifact_dir);
    let config = if fresh || !artifact_dir_path.exists() {
        create_artifact_dir(artifact_dir);
        config
            .save(format!("{artifact_dir}/config.json"))
            .expect("Failed to save config.json");
        config
    } else {
        TrainingConfig::load(format!("{artifact_dir}/config.json"))
            .expect("Failed to load config file.")
    };

    B::seed(config.seed);

    let batcher_train = DatasetBatcher::<B>::new(device.clone());
    let batcher_valid = DatasetBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Data::train());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Data::valid());

    let mut learner_builder = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary();

    if !fresh {
        let epoch = get_last_epoch(artifact_dir);
        learner_builder = learner_builder.checkpoint(epoch);
    }

    let learner = learner_builder.build(
        config.model.init::<B>(&device),
        config.optimizer.init(),
        config.learning_rate,
    );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");
}
