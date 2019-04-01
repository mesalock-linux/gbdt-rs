//! This module implements the data loader.
//!
//! Currently we support to kind of input format: csv format and libsvm data format.
//!
//! # Example
//! ## LibSVM format
//! ```rust
//! use gbdt::input::InputFormat;
//! use gbdt::input;
//! let test_file = "data/xgb_binary_logistic/agaricus.txt.test";
//! let mut fmt = input::InputFormat::txt_format();
//! fmt.set_feature_size(126);
//! fmt.set_delimeter(' ');
//! let test_data = input::load(test_file, fmt);
//! ```
//!
//! ## CSV format
//! ```rust
//! use gbdt::input::InputFormat;
//! use gbdt::input;
//! let test_file = "data/xgb_multi_softmax/dermatology.data.test";
//! let mut fmt = InputFormat::csv_format();
//! fmt.set_feature_size(34);
//! let test_data = input::load(test_file, fmt);
//! ```

use crate::decision_tree::{Data, DataVec, ValueType, VALUE_TYPE_UNKNOWN};

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

use regex::Regex;
use serde_derive::{Deserialize, Serialize};

/// This enum type defines the data file format.
///
/// We support two data format:
/// 1. CSV format
/// 2. LibSVM data format
#[derive(Copy, Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    /// CSV format
    CSV,
    /// LibSVM data format
    TXT,
}

/// The input file format struct.
#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
pub struct InputFormat {
    /// Data file format
    pub ftype: FileFormat,

    /// Set if ftype is set to [FileFormat](enum.FileFormat.CSV).
    /// Indicates whether the csv has header.
    pub header: bool,

    /// Set if ftype is set to [FileFormat](enum.FileFormat.CSV).
    /// Indicates which colume is the data label. (default = 0)
    pub label_idx: usize,

    /// Set if ftype is set to [FileFormat](enum.FileFormat.CSV).
    /// Indicates if we allow unknown value in data file or not.
    pub enable_unknown_value: bool,

    /// Delimeter of the data file.
    pub delimeter: char,

    /// Set if ftype is set to [FileFormat](enum.FileFormat.TXT).
    /// Indicates the total feature size.
    pub feature_size: usize,
}

impl InputFormat {
    /// Return a default CSV input format.
    /// # Example
    /// ```rust
    /// use gbdt::input::InputFormat;
    /// let mut fmt = InputFormat::csv_format();
    /// println!("{}", fmt.to_string());
    /// ```
    pub fn csv_format() -> InputFormat {
        InputFormat {
            ftype: FileFormat::CSV,
            header: false,
            label_idx: 0,
            enable_unknown_value: false,
            delimeter: ',',
            feature_size: 0,
        }
    }

    /// Return a default LibSVM input format.
    /// # Example
    /// ```rust
    /// use gbdt::input::InputFormat;
    /// let mut fmt = InputFormat::txt_format();
    /// println!("{}", fmt.to_string());
    /// ```
    pub fn txt_format() -> InputFormat {
        InputFormat {
            ftype: FileFormat::TXT,
            header: false,
            label_idx: 0,
            enable_unknown_value: false,
            delimeter: '\t',
            feature_size: 0,
        }
    }
    /// Transform the input format to human readable string.
    /// # Example
    /// ```rust
    /// use gbdt::input::InputFormat;
    /// let mut fmt = InputFormat::csv_format();
    /// println!("{}", fmt.to_string());
    /// ```
    pub fn to_string(&self) -> String {
        let mut s = String::from("");
        s.push_str(&format!(
            "File type: {}\n",
            match self.ftype {
                FileFormat::CSV => "CSV",
                FileFormat::TXT => "TXT",
            }
        ));
        match self.ftype {
            FileFormat::CSV => {
                s.push_str(&format!("Has header: {}\n", self.header));
                s.push_str(&format!("Label index: {}\n", self.label_idx));
            }
            FileFormat::TXT => {
                s.push_str(&format!("Feature size: {}\n", self.feature_size));
            }
        }
        s.push_str(&format!("Delemeter: [{}]", self.delimeter));
        s
    }

    /// Set feature size for the LibSVM input format.
    /// # Example
    /// ```rust
    /// use gbdt::input::InputFormat;
    /// let mut fmt = InputFormat::txt_format();
    /// fmt.set_feature_size(126); // the total feature size
    /// ```
    pub fn set_feature_size(&mut self, size: usize) {
        self.feature_size = size;
    }

    /// Set for label index for CSV format.
    /// # Example
    /// ```rust
    /// use gbdt::input::InputFormat;
    /// let mut fmt = InputFormat::csv_format();
    /// fmt.set_label_index(34);
    /// ```
    pub fn set_label_index(&mut self, idx: usize) {
        self.label_idx = idx;
    }

    /// Set for label index for CSV format.
    /// # Example
    /// ```rust
    /// use gbdt::input::InputFormat;
    /// let mut fmt = InputFormat::txt_format();
    /// fmt.set_delimeter(' ');
    /// ```
    pub fn set_delimeter(&mut self, delim: char) {
        self.delimeter = delim;
    }
}

/// Function for char counting, used in [infer](function.input.infer)
fn count(mut hash_map: HashMap<char, u32>, word: char) -> HashMap<char, u32> {
    {
        let c = hash_map.entry(word).or_insert(0);
        *c += 1;
    }
    hash_map
}

/// Function used for input file type inference. This can help recognize the file format.
/// If the file is in csv type, this function also helps to check whether the csv file has
/// header.
///
/// # Example
/// ```rust
/// use gbdt::input::infer;
/// let train_file = "dataset/iris/train.txt";
/// let fmt = infer(train_file);
/// println!("{}", fmt.to_string());
/// ```
pub fn infer(file_name: &str) -> InputFormat {
    let file = File::open(file_name.to_string()).unwrap();
    let mut reader = BufReader::new(file);

    // check CSV or TXT
    let mut first_line = String::new();
    reader.read_line(&mut first_line).unwrap();
    let mut input_format = if first_line.contains(':') {
        InputFormat::txt_format()
    } else {
        InputFormat::csv_format()
    };

    // Check delimeter
    let reg = match input_format.ftype {
        FileFormat::CSV => Regex::new(r"[+-]?\d+(,\d+)*(.\d+(e\d+)?)?").unwrap(),
        FileFormat::TXT => Regex::new(r"\d+:[+-]?\d+(,\d+)*(.\d+(e\d+)?)?").unwrap(),
    };
    let mut second_line = String::new();
    reader
        .read_line(&mut second_line)
        .expect("No second line to read");
    let caps = reg.captures(&second_line).unwrap().len();
    let second_line_after = reg.replace_all(&second_line, "");
    let cnt = second_line_after.chars().fold(HashMap::new(), count);
    let default_delim: char = match input_format.ftype {
        FileFormat::CSV => ',',
        FileFormat::TXT => '\t',
    };
    let mut flag = false;
    if let Some(value) = cnt.get(&default_delim) {
        if *value > ((caps as u32) - 2) {
            input_format.delimeter = default_delim;
            flag = true;
        }
    }
    if !flag {
        let mut max_cnt: u32 = 0;
        let mut delim = '\t';
        for (k, v) in &cnt {
            if *v > max_cnt {
                max_cnt = *v;
                delim = *k;
            }
        }
        input_format.delimeter = delim;
        flag = true;
    }
    // we shouldn't reach here
    assert_eq!(flag, true);

    // if CSV, check header
    // use the first value as label or target value by default
    match input_format.ftype {
        FileFormat::CSV => {
            let first_line_after = reg.replace_all(&first_line, "");
            let letters = Regex::new(r"[a-zA-Z]").unwrap();
            match letters.captures(&first_line_after) {
                Some(letter_caps) => {
                    input_format.header = letter_caps.len() > 0;
                }
                None => {}
            }
        }
        _ => {}
    };

    input_format
}

/// Load csv file.
/// # Example
/// ```rust
/// use std::fs::File;
/// use gbdt::input::{InputFormat, load_csv};
/// let train_file = "dataset/iris/train.txt";
/// let mut file = File::open(train_file.to_string()).unwrap();
/// let mut fmt = InputFormat::csv_format();
/// fmt.set_label_index(4);
/// let train_dv = load_csv(&mut file, fmt);
/// ```
///
/// # Error
/// Raise error if file cannot be read correctly.
pub fn load_csv(file: &mut File, input_format: InputFormat) -> Result<DataVec, Box<Error>> {
    file.seek(SeekFrom::Start(0))?;
    let mut dv = Vec::new();

    let mut reader = BufReader::new(file);
    let mut l = String::new();
    if input_format.header {
        reader.read_line(&mut l).unwrap_or(0);
    }
    let mut v: Vec<ValueType>;
    for line in reader.lines() {
        let content = line?;
        if input_format.enable_unknown_value {
            v = content
                .split(input_format.delimeter)
                .map(|x| x.parse::<ValueType>().unwrap_or(VALUE_TYPE_UNKNOWN))
                .collect();
        } else {
            v = content
                .split(input_format.delimeter)
                .map(|x| x.parse::<ValueType>().unwrap())
                .collect();
        }
        dv.push(Data {
            label: v.swap_remove(input_format.label_idx),
            feature: v,
            target: 0.0,
            weight: 1.0,
            residual: 0.0,
            initial_guess: 0.0,
        })
    }
    Ok(dv)
}

/// Load txt file.
///
/// # Example
/// ```rust
/// use std::fs::File;
/// use gbdt::input::{InputFormat, load_txt};
/// let test_file = "xgb-data/xgb_binary_logistic/agaricus.txt.test";
/// let mut file = File::open(test_file.to_string()).unwrap();
/// let mut fmt = InputFormat::csv_format();
/// fmt.set_feature_size(126);
/// fmt.set_delimeter(' ');
/// let test_dv = load_txt(&mut file, fmt);
/// ```
///
/// # Error
/// Raise error if file cannot be read correctly.
pub fn load_txt(file: &mut File, input_format: InputFormat) -> Result<DataVec, Box<Error>> {
    file.seek(SeekFrom::Start(0))?;
    let mut dv = Vec::new();

    let reader = BufReader::new(file);
    let mut label: ValueType = 0.0;
    let mut idx: usize = 0;
    let mut val: ValueType = 0.0;
    for line in reader.lines() {
        let mut v: Vec<ValueType> = vec![VALUE_TYPE_UNKNOWN; input_format.feature_size];
        for token in line.unwrap().split(input_format.delimeter) {
            let splited_token: Vec<&str> = token.split(':').collect();
            if splited_token.len() == 2 {
                let mut err = false;
                match splited_token[0 as usize].parse::<usize>() {
                    Ok(kk) => {
                        idx = kk;
                    }
                    Err(_) => err = true,
                }
                match splited_token[1 as usize].parse::<ValueType>() {
                    Ok(vv) => {
                        val = vv;
                    }
                    Err(_) => err = true,
                }
                if idx >= input_format.feature_size {
                    err = true;
                }
                if !err {
                    v[idx] = val;
                }
            }
            if splited_token.len() == 1 {
                label = splited_token[0 as usize].parse::<ValueType>().unwrap();
            } else {
                // report error
            }
        }
        dv.push(Data {
            label,
            feature: v,
            target: 0.0,
            weight: 1.0,
            residual: 0.0,
            initial_guess: 0.0,
        });
    }

    Ok(dv)
}

/// Load file with certain input format.
/// # Example
/// ## LibSVM format
/// ```rust
/// use gbdt::input::InputFormat;
/// use gbdt::input;
/// let test_file = "data/xgb_binary_logistic/agaricus.txt.test";
/// let mut fmt = input::InputFormat::txt_format();
/// fmt.set_feature_size(126);
/// fmt.set_delimeter(' ');
/// let test_data = input::load(test_file, fmt);
/// ```
///
/// ## CSV format
/// ```rust
/// use gbdt::input::InputFormat;
/// use gbdt::input;
/// let test_file = "data/xgb_multi_softmax/dermatology.data.test";
/// let mut fmt = InputFormat::csv_format();
/// fmt.set_feature_size(34);
/// let test_data = input::load(test_file, fmt);
/// ```
///
/// # Error
/// Raise error if file cannot be open correctly.
pub fn load(file_name: &str, input_format: InputFormat) -> Result<DataVec, Box<Error>> {
    let mut file = File::open(file_name.to_string())?;
    match input_format.ftype {
        FileFormat::CSV => load_csv(&mut file, input_format),
        FileFormat::TXT => load_txt(&mut file, input_format),
    }
}
