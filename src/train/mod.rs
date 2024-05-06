use std::path::Path;

pub mod classification;
pub mod regression;

pub(crate) fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub(crate) fn get_last_epoch(artifact_dir: &str) -> usize {
    let mut max_epoch = 0;
    for entry in std::fs::read_dir(Path::new(&format!("{artifact_dir}/train")))
        .expect("Failed to read artifact directory")
    {
        let file_name = entry.expect("Failed to read entry").file_name();
        let file_name_str = file_name.to_string_lossy();

        if let Some(epoch_str) = file_name_str.strip_prefix("epoch-") {
            if let Ok(epoch) = epoch_str.parse::<usize>() {
                if epoch > max_epoch {
                    max_epoch = epoch;
                }
            }
        }
    }

    max_epoch
}
