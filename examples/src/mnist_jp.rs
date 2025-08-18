use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

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
                print!("*");
            } else {
                print!("o");
            }
        }
        println!();
    }
}

const IMAGE_BYTES: usize = 512;

pub struct Preproc;

impl Preproc {
    pub fn get_images(path_str: &str) -> Vec<Image> {
        let paths = fs::read_dir(path_str).unwrap();
        let mut images: Vec<[u8; IMAGE_BYTES]> = Vec::new();
        let mut images_copy: Vec<Image> = Vec::new();

        for path in paths {
            let mut idx_bytes: usize = 0;

            for i in 0..20 {
                let mut buf = [0; IMAGE_BYTES];
                let mut file = match File::open(&path.unwrap().path()) {
                    Err(why) => panic!("couldn't open the File: {}", why),
                    Ok(file) => file,
                };
                let _ = file.seek(SeekFrom::Start(idx_bytes as u64));
                let _ = file.read_exact(&mut buf);

                images.push(buf);

                idx_bytes += IMAGE_BYTES;
            }
        }

        for image in &images {
            let mut buf = [[0u8; 64]; 64];

            for y in 0..64 {
                for x in 0..8 {
                    let mut mask = 0x80u8;

                    for i in 0..8 {
                        if (image[y * 8 + x] & mask) == 0 {
                            buf[y][x * 8 + i] |= 0;
                        } else {
                            buf[y][x * 8 + i] |= 1;
                        }
                        mask = mask >> 1;
                    }
                }
            }

            let mut image_copy: Image = Image::new();

            for y in 0..64 {
                for x in 0..64 {
                    image_copy.data.push(buf[y][x]);
                }
            }
            image_copy.blocksize = 64;
            images_copy.push(image_copy);
        }

        return images_copy;
    }

    pub fn normalize(image: Image) -> Image {
        let img = Image::to_array(image);
        let mut it: usize = 63;
        let mut ib: usize = 0;
        let mut il: usize = 63;
        let mut ir: usize = 0;

        for y in 0..64 {
            for x in 0..64 {
                if img[y][x] == 1 {
                    ib = if ib < y { y } else { ib };
                    it = if it > y { y } else { it };
                    il = if il > x { x } else { il };
                    ir = if ir < x { x } else { ir };
                }
            }
        }

        let _blocksize = ir - il;
        let mut img_copy = Image::new();

        for y in it..ib + 1 {
            for x in il..ir + 1 {
                img_copy.data.push(img[y][x]);
            }
        }

        img_copy.blocksize = _blocksize + 1;

        return img_copy;
    }
}

fn main() {
    let images: Vec<Image> = Preproc::get_images("./data/MNIST_JP");

    for img in images {
        img.display();

        break;
    }
}
