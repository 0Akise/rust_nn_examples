use gpu_accel::{Shape, Tensor};
use nn_backbone::autograd::gpu_context::GpuContext;
use nn_backbone::autograd::variable::Variable;
use nn_backbone::layer::layers;
use nn_backbone::layer::model::Sequential;
use nn_backbone::layer::train::SequentialTrainer;

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::Arc;

use tokio::sync::Mutex;

const MAGIC_IMAGE: u32 = 2051;
const MAGIC_LABEL: u32 = 2049;
const SIZE: u32 = 28;

#[derive(Debug, Clone)]
pub struct Image {
    pub pixels: Vec<u8>,
    pub label: u8,
}

impl Image {
    pub fn new(pixels: Vec<u8>, label: u8) -> Self {
        return Self { pixels, label };
    }

    pub fn to_normalized(&self) -> Vec<f32> {
        return self.pixels.iter().map(|&x| x as f32 / 255.0).collect();
    }

    pub fn display(&self) {
        println!("Label: {}", self.label);

        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                let pixel = self.pixels[y * 28 + x];
                let char = match pixel {
                    0..=63 => " ",
                    64..=127 => "·",
                    128..=191 => "▫",
                    192..=255 => "█",
                };
                print!("{}", char);
            }
            println!();
        }
    }

    pub fn to_one_hot(&self) -> Vec<f32> {
        let mut one_hot = vec![0.0; 10];

        one_hot[self.label as usize] = 1.0;

        return one_hot;
    }
}

#[derive(Debug)]
pub struct Dataset {
    pub images: Vec<Image>,
}

impl Dataset {
    pub fn new() -> Self {
        return Self { images: Vec::new() };
    }

    fn read_u32_be(reader: &mut BufReader<File>) -> Result<u32, std::io::Error> {
        let mut bytes = [0u8; 4];

        reader.read_exact(&mut bytes).unwrap();

        return Ok(u32::from_be_bytes(bytes));
    }

    fn load_images(file_path: &str) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        let mut file = BufReader::new(File::open(file_path).unwrap());
        let magic = Self::read_u32_be(&mut file).unwrap();
        let num_images = Self::read_u32_be(&mut file).unwrap();
        let num_rows = Self::read_u32_be(&mut file).unwrap();
        let num_cols = Self::read_u32_be(&mut file).unwrap();

        if magic != MAGIC_IMAGE {
            return Err(format!("Invalid magic number for images: {}", magic).into());
        }

        if num_rows != SIZE || num_cols != SIZE {
            return Err(format!("Expected 28x28 images, got {}x{}", num_rows, num_cols).into());
        }

        let mut images = Vec::with_capacity(num_images as usize);

        for _ in 0..num_images {
            let mut pixels = vec![0u8; 784];

            file.read_exact(&mut pixels).unwrap();
            images.push(pixels);
        }

        return Ok(images);
    }

    fn load_labels(file_path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut file = BufReader::new(File::open(file_path).unwrap());
        let magic = Self::read_u32_be(&mut file).unwrap();
        let num_labels = Self::read_u32_be(&mut file).unwrap();

        if magic != MAGIC_LABEL {
            return Err(format!("Invalid magic number for labels: {}", magic).into());
        }

        let mut labels = vec![0u8; num_labels as usize];

        file.read_exact(&mut labels).unwrap();

        return Ok(labels);
    }

    pub fn load_from_files(images_path: &str, labels_path: &str) -> Result<Self, Box<dyn Error>> {
        let images_data = Self::load_images(images_path).unwrap();
        let labels_data = Self::load_labels(labels_path).unwrap();

        if images_data.len() != labels_data.len() {
            return Err(format!(
                "Mismatch: {} images but {} labels",
                images_data.len(),
                labels_data.len()
            )
            .into());
        }

        let mut dataset = Self::new();

        for (pixels, label) in images_data.into_iter().zip(labels_data.into_iter()) {
            dataset.images.push(Image::new(pixels, label));
        }

        println!("Loaded {} MNIST samples ✅", dataset.images.len());

        return Ok(dataset);
    }

    pub fn split_train_valid(&self, train_ratio: f32) -> (Dataset, Dataset) {
        let split_index = (self.images.len() as f32 * train_ratio) as usize;
        let mut train_set = Dataset::new();
        let mut valid_set = Dataset::new();

        for (i, image) in self.images.iter().enumerate() {
            if i < split_index {
                train_set.images.push(image.clone());
            } else {
                valid_set.images.push(image.clone());
            }
        }

        (train_set, valid_set)
    }

    pub fn to_tensor_data(&self) -> (Vec<f32>, Vec<f32>) {
        let mut data_image = Vec::with_capacity(self.images.len() * 784);
        let mut data_label = Vec::with_capacity(self.images.len() * 10);

        for image in &self.images {
            data_image.extend(image.to_normalized());
            data_label.extend(image.to_one_hot());
        }

        (data_image, data_label)
    }

    pub fn print_stats(&self) {
        println!("\nDataset 📊:");
        println!("  Total samples: {}", self.images.len());
    }
}

pub struct MNISTLoader;

impl MNISTLoader {
    fn load_train_data() -> Result<Dataset, Box<dyn Error>> {
        return Dataset::load_from_files(
            "./data/MNIST/train-images.idx3-ubyte",
            "./data/MNIST/train-labels.idx1-ubyte",
        );
    }

    fn load_test_data() -> Result<Dataset, Box<dyn Error>> {
        return Dataset::load_from_files(
            "./data/MNIST/test-images.idx3-ubyte",
            "./data/MNIST/test-labels.idx1-ubyte",
        );
    }

    pub fn load() -> Result<(Dataset, Dataset), Box<dyn Error>> {
        let train = Self::load_train_data().unwrap();
        let test = Self::load_test_data().unwrap();

        return Ok((train, test));
    }
}

pub struct MNISTClassifier {
    model: Option<Sequential>,
    ctx: Arc<Mutex<GpuContext>>,
}

impl MNISTClassifier {
    pub async fn build_context() -> Result<Self, Box<dyn Error>> {
        let ctx = Arc::new(Mutex::new(GpuContext::new().await?));

        return Ok(Self { model: None, ctx });
    }

    pub async fn with_model(mut self) -> Result<Self, Box<dyn Error>> {
        let ctx = Arc::clone(&self.ctx);

        let model = Sequential::new()
            .add(layers::linear(784, 128, ctx.clone()).await?)
            .add(layers::relu())
            .add(layers::linear(128, 64, ctx.clone()).await?)
            .add(layers::relu())
            .add(layers::linear(64, 10, ctx.clone()).await?)
            .add(layers::softmax());

        self.model = Some(model);

        return Ok(self);
    }

    pub async fn forward(&mut self, batch: &[&Image]) -> Result<Vec<usize>, Box<dyn Error>> {
        let model = self.model.as_mut().ok_or("Model not initialized")?;
        let batch_size = batch.len();

        if batch_size == 0 {
            return Err("Empty batch provided".into());
        }

        println!("Forward pass for batch size: {}", batch_size);

        let mut image_data = Vec::with_capacity(batch_size * 784);
        for image in batch {
            image_data.extend(image.to_normalized());
        }

        if image_data.len() != batch_size * 784 {
            return Err(format!(
                "Data size mismatch: expected {}, got {}",
                batch_size * 784,
                image_data.len()
            )
            .into());
        }

        let input = Variable::with_grad(Tensor::new(image_data, Shape::new(vec![batch_size, 784])));

        println!("Input shape: {:?}", input.shape());

        let output = model.forward(&input).await?;

        println!("Output shape: {:?}", output.shape());

        let mut predictions = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start_idx = i * 10;
            let end_idx = start_idx + 10;
            let sample_output = &output.tensor.data[start_idx..end_idx];

            let predicted_class = sample_output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            predictions.push(predicted_class);
        }

        return Ok(predictions);
    }

    pub fn calculate_batch_size(total_samples: usize) -> usize {
        const MAX_ELEMENTS_PER_BATCH: usize = 1_000_000;
        const MNIST_FEATURES: usize = 784;

        let max_batch = MAX_ELEMENTS_PER_BATCH / MNIST_FEATURES;
        let desired_iterations = 12;
        let computed_batch = (total_samples + desired_iterations - 1) / desired_iterations;

        computed_batch.min(max_batch).max(100)
    }

    pub async fn evaluate(
        &mut self,
        dataset: &Dataset,
        max_samples: Option<usize>,
    ) -> Result<f32, Box<dyn Error>> {
        let total = max_samples
            .unwrap_or(dataset.images.len())
            .min(dataset.images.len());

        const SAFE_ITERATION_CHUNK: usize = 5;

        let batch_size = if total <= 1000 {
            200
        } else if total <= 5000 {
            500
        } else {
            (total / SAFE_ITERATION_CHUNK).min(1000).max(200)
        };

        let iterations = (total + batch_size - 1) / batch_size;

        println!(
            "Evaluating {} samples with batch size {} (total {} iterations)",
            total, batch_size, iterations
        );
        println!(
            "  Working in chunks of {} iterations to avoid GPU overflow",
            SAFE_ITERATION_CHUNK
        );

        let mut correct = 0;
        let mut iterations_since_reset = 0;

        for batch_idx in 0..iterations {
            if iterations_since_reset >= SAFE_ITERATION_CHUNK && batch_idx < iterations - 1 {
                println!(
                    "  ⚠️  Approaching GPU limit after {} iterations, forcing synchronization...",
                    iterations_since_reset
                );

                {
                    let ctx = self.ctx.lock().await;
                    let gpu = ctx.gpu().lock().await;

                    for _ in 0..3 {
                        let _ = gpu.device.poll(wgpu::PollType::Wait);
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                }

                iterations_since_reset = 0;
                println!("  ✅ GPU synchronized, continuing...");
            }

            let batch_start = batch_idx * batch_size;
            let batch_end = ((batch_idx + 1) * batch_size).min(total);

            if batch_start >= total || batch_end <= batch_start {
                continue;
            }

            let current_batch: Vec<&Image> =
                dataset.images[batch_start..batch_end].iter().collect();

            println!(
                "  Processing batch {}-{} (iteration {}/{}, chunk position {})",
                batch_start + 1,
                batch_end,
                batch_idx + 1,
                iterations,
                iterations_since_reset + 1
            );

            let predictions = self.forward(&current_batch).await?;

            for (i, &predicted) in predictions.iter().enumerate() {
                if predicted == (current_batch[i].label as usize) {
                    correct += 1;
                }
            }

            iterations_since_reset += 1;

            {
                let ctx = self.ctx.lock().await;
                let gpu = ctx.gpu().lock().await;
                let _ = gpu.device.poll(wgpu::PollType::Wait);
            }
        }

        let accuracy = correct as f32 / total as f32;
        println!(
            "Accuracy: {:.2}% ({}/{} correct)",
            accuracy * 100.0,
            correct,
            total
        );

        Ok(accuracy)
    }

    pub async fn train(
        &mut self,
        train_set: &Dataset,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(), Box<dyn Error>> {
        println!(
            "🎓 Starting training with {} epochs, batch size {}",
            epochs, batch_size
        );

        let model = self.model.take().unwrap();
        let mut trainer = SequentialTrainer::new(model, 0.01, self.ctx.clone());

        for epoch in 0..epochs {
            println!("\n📚 Epoch {}/{}", epoch + 1, epochs);

            let mut epoch_loss = 0.0;
            let num_batches = (train_set.images.len() + batch_size - 1) / batch_size;
            let mut processed_batches = 0;

            for batch_idx in 0..num_batches {
                if processed_batches > 0 && processed_batches % 3 == 0 {
                    println!("  🔄 GPU sync after {} batches", processed_batches);
                    let ctx = self.ctx.lock().await;
                    let gpu = ctx.gpu().lock().await;
                    let _ = gpu.device.poll(wgpu::PollType::Wait);
                    drop(gpu);
                    drop(ctx);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }

                let start = batch_idx * batch_size;
                let end = ((batch_idx + 1) * batch_size).min(train_set.images.len());

                let batch_images: Vec<f32> = train_set.images[start..end]
                    .iter()
                    .flat_map(|img| img.to_normalized())
                    .collect();

                let batch_labels: Vec<f32> = train_set.images[start..end]
                    .iter()
                    .flat_map(|img| img.to_one_hot())
                    .collect();

                let batch_inputs = Variable::with_grad(Tensor::new(
                    batch_images,
                    Shape::new(vec![end - start, 784]),
                ));

                let batch_targets = Variable::with_grad(Tensor::new(
                    batch_labels,
                    Shape::new(vec![end - start, 10]),
                ));

                println!(
                    "  Batch {}/{}: samples {}-{}",
                    batch_idx + 1,
                    num_batches,
                    start + 1,
                    end
                );

                let loss = trainer.train_step(&batch_inputs, &batch_targets).await?;
                epoch_loss += loss;
                processed_batches += 1;

                println!("    Loss: {:.4}", loss);
            }

            println!(
                "  Epoch {} complete - Average Loss: {:.4}",
                epoch + 1,
                epoch_loss / num_batches as f32
            );
        }

        self.model = Some(trainer.model);
        Ok(())
    }
}

async fn demo() -> Result<(), Box<dyn Error>> {
    println!("MNIST Neural Network Demo");
    println!();

    let (train, test) = MNISTLoader::load().unwrap();
    let mut classifier = MNISTClassifier::build_context().await?.with_model().await?;
    let test_samples = 1000;
    let accuracy_init = classifier.evaluate(&test, Some(test_samples)).await?;

    println!("\nModel summary 📈:");
    println!(
        "  Randomized weight accuracy:   {:.1}%",
        accuracy_init * 100.0
    );

    let train_epochs = 5;
    let train_subset = 2000;
    let batch_size = 100;
    let mut small_train = Dataset::new();

    small_train.images = train.images.into_iter().take(train_subset).collect();

    let (train_set, valid_set) = small_train.split_train_valid(0.8);

    println!(
        "\nTraining on {} samples, validating on {} samples",
        train_set.images.len(),
        valid_set.images.len()
    );

    classifier
        .train(&train_set, train_epochs, batch_size)
        .await?;

    let accuracy_learned = classifier.evaluate(&test, Some(1000)).await?;

    println!("\nModel summary 📈:");
    println!("  Initial: {:.1}%", accuracy_init * 100.0);
    println!("  Final:   {:.1}%", accuracy_learned * 100.0);

    return Ok(());
}

#[tokio::main]
async fn main() {
    pollster::block_on(async {
        if let Err(e) = demo().await {
            println!("Error: {}", e);
        }
    });
}
