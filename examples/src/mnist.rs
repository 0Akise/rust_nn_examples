use std::fs::File;
use std::io::{BufReader, Read};

use gpu_accel::{GpuSession, Shape, Tensor};

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

    fn load_images(file_path: &str) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
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

    fn load_labels(file_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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

    pub fn load_from_files(
        images_path: &str,
        labels_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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

        println!("✅ Loaded {} MNIST samples", dataset.images.len());

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
        let mut image_data = Vec::with_capacity(self.images.len() * 784);
        let mut label_data = Vec::with_capacity(self.images.len() * 10);

        for image in &self.images {
            image_data.extend(image.to_normalized());
            label_data.extend(image.to_one_hot());
        }

        (image_data, label_data)
    }

    pub fn print_stats(&self) {
        println!("\n📊 Dataset:");
        println!("  Total samples: {}", self.images.len());
    }
}

pub struct MNISTLoader;

impl MNISTLoader {
    pub fn load_train_data() -> Result<Dataset, Box<dyn std::error::Error>> {
        return Dataset::load_from_files(
            "./data/MNIST/train-images.idx3-ubyte",
            "./data/MNIST/train-labels.idx1-ubyte",
        );
    }

    pub fn load_test_data() -> Result<Dataset, Box<dyn std::error::Error>> {
        return Dataset::load_from_files(
            "./data/MNIST/train-images.idx3-ubyte",
            "./data/MNIST/train-labels.idx1-ubyte",
        );
    }

    pub fn load() -> Result<(Dataset, Dataset), Box<dyn std::error::Error>> {
        let train = Self::load_train_data().unwrap();
        let test = Self::load_test_data().unwrap();

        return Ok((train, test));
    }
}

pub struct MNISTTrainer {
    session: GpuSession,
}

impl MNISTTrainer {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        return Ok(Self {
            session: GpuSession::new().await.unwrap(),
        });
    }

    pub fn batch_to_tensors(&self, batch: &[&Image]) -> (Tensor, Tensor) {
        let batch_size = batch.len();
        let mut image_data = Vec::with_capacity(batch_size * 784);
        let mut label_data = Vec::with_capacity(batch_size * 10);

        for image in batch {
            image_data.extend(image.to_normalized());
            label_data.extend(image.to_one_hot());
        }

        let tensor_images = Tensor::new(image_data, Shape::new(vec![batch_size, 784]));
        let tensor_labels = Tensor::new(label_data, Shape::new(vec![batch_size, 10]));

        (tensor_images, tensor_labels)
    }

    pub async fn forward(&mut self, dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
        let batch = dataset.images.iter().take(32).collect::<Vec<_>>();
        let (input, labels) = self.batch_to_tensors(&batch);
        let weights = Tensor::new(vec![0.01; 784 * 10], Shape::new(vec![784, 10]));
        let output = self.session.matmul(&input, &weights).await?;

        println!("forward pass:");
        println!("    Input shape: {:?}", input.shape.dims);
        println!("    Weights shape: {:?}", weights.shape.dims);
        println!("    Output shape: {:?}", output.shape.dims);
        println!("    First prediction: {:?}", &output.data[0..10]);

        return Ok(());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train, test) = MNISTLoader::load().unwrap();

    println!("\n    Sample images:");

    for i in 0..3 {
        if let Some(sample) = train.images.get(i) {
            sample.display();

            println!();
        }
    }

    let mut trainer = pollster::block_on(MNISTTrainer::new()).unwrap();

    pollster::block_on(trainer.forward(&train)).unwrap();

    return Ok(());
}
