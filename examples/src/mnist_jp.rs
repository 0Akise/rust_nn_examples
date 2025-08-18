use std::fs::{self, File};
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub struct DatasetStats {
    pub total_images: usize,
    pub total_pixels: usize,
    pub density: f32,
}

#[derive(Clone)]
pub struct Image {
    pub data: Vec<u8>,
    pub blocksize: usize,
}

impl Image {
    pub fn new() -> Self {
        return Self {
            data: Vec::new(),
            blocksize: 0,
        };
    }

    pub fn to_array(target: Image) -> [[u8; 64]; 64] {
        let mut array = [[0u8; 64]; 64];

        for y in 0..64 {
            for x in 0..64 {
                array[y][x] = target.data[y * 64 + x];
            }
        }

        return array;
    }

    pub fn to_image(target: [[u8; 64]; 64]) -> Image {
        let mut image: Image = Image::new();

        for y in 0..64 {
            for x in 0..64 {
                image.data.push(target[y][x]);
            }
        }

        return image;
    }

    pub fn display(&self) {
        for idx in 0..self.data.len() {
            if (idx % self.blocksize == 0) && (idx != 0) {
                println!();
            }
            if self.data[idx] == 1 {
                print!("█");
            } else {
                print!("·");
            }
        }
        println!();
    }
}

const IMAGE_BYTES: usize = 512;

pub struct Preproc;

impl Preproc {
    fn unpack_image(packed_data: [u8; IMAGE_BYTES]) -> Image {
        let mut image = Image::new();

        image.data.reserve_exact(64 * 64);

        for y in 0..64 {
            for x_byte in 0..8 {
                let byte = packed_data[y * 8 + x_byte];

                for bit_pos in 0..8 {
                    let pixel = if (byte & (0x80 >> bit_pos)) != 0 {
                        1
                    } else {
                        0
                    };

                    image.data.push(pixel);
                }
            }
        }

        image.blocksize = 64;

        return image;
    }

    fn load_images_from_file(file_path: &Path) -> std::io::Result<Vec<Image>> {
        let file = File::open(file_path).unwrap();
        let file_size = file.metadata().unwrap().len() as usize;

        if file_size % IMAGE_BYTES != 0 {
            eprintln!(
                "⚠️  Warning: File size {} is not multiple of image size {}",
                file_size, IMAGE_BYTES
            );
        }

        let images_count = file_size / IMAGE_BYTES;
        let mut images = Vec::with_capacity(images_count);
        let mut reader = BufReader::new(file);

        for _ in 0..images_count {
            let mut buffer = [0u8; IMAGE_BYTES];

            reader.read_exact(&mut buffer).unwrap();

            let image = Self::unpack_image(buffer);

            images.push(image);
        }

        return Ok(images);
    }

    pub fn get_images(path_str: &str) -> Result<Vec<Image>, Box<dyn std::error::Error>> {
        let dir_path = Path::new(path_str);
        let entries: Vec<_> = fs::read_dir(dir_path)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_file())
            .collect();
        let mut all_images = Vec::new();
        let mut total_files = 0;
        let mut total_images = 0;

        for entry in entries {
            let file_path = entry.path();

            match Self::load_images_from_file(&file_path) {
                Ok(mut images) => {
                    total_files += 1;
                    total_images += images.len();
                    all_images.append(&mut images);
                }

                Err(e) => {
                    eprintln!("⚠️  Failed to load {:?}: {}", file_path, e);
                }
            }
        }

        println!(
            "Summary: {} images from {} files",
            total_images, total_files
        );

        return Ok(all_images);
    }

    pub fn images_to_tensor_data(images: &[Image]) -> Vec<f32> {
        images
            .iter()
            .flat_map(|img| img.data.iter().map(|&x| x as f32))
            .collect()
    }

    pub fn analyze_dataset(images: &[Image]) -> DatasetStats {
        let total_pixels: usize = images.iter().map(|img| img.data.len()).sum();
        let total_ones: usize = images
            .iter()
            .map(|img| img.data.iter().filter(|&&x| x == 1).count())
            .sum();

        return DatasetStats {
            total_images: images.len(),
            total_pixels,
            density: total_ones as f32 / total_pixels as f32,
        };
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images = Preproc::get_images("./data/MNIST_JP/images").unwrap();

    if let Some(first_image) = images.first() {
        println!("First image:");

        first_image.display();
    }

    let stats = Preproc::analyze_dataset(&images);

    println!("\nDataset Statistics:");
    println!("  Images: {}", stats.total_images);
    println!("  Total pixels: {}", stats.total_pixels);
    println!("  Density: {:.2}%", stats.density * 100.0);

    let tensor_data = Preproc::images_to_tensor_data(&images);

    println!("  Tensor data length: {}", tensor_data.len());

    return Ok(());
}
