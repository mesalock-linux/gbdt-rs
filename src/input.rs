use crate::decision_tree::{ValueType, Data, DataVec};

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::collections::HashMap;

extern crate regex;
use regex::Regex;

#[derive(Copy, Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    CSV,
    TXT,
}

#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
pub struct InputFormat {
    pub ftype: FileFormat,
    pub header: bool,
    pub label_idx: usize,
    pub delimeter: char,
    pub feature_size: usize,
}

impl InputFormat {
    pub fn csv_format() -> InputFormat {
        InputFormat {
            ftype: FileFormat::CSV,
            header: false,
            label_idx: 0,
            delimeter: ',',
            feature_size: 0,
        }
    }

    pub fn txt_format() -> InputFormat {
        InputFormat {
            ftype: FileFormat::TXT,
            header: false,
            label_idx: 0,
            delimeter: '\t',
            feature_size: 0,
        }
    }

    pub fn to_string(&self) -> String {
        let mut s = String::from("");
        s.push_str(&format!("File type: {}\n", match self.ftype {
            FileFormat::CSV => { "CSV" },
            FileFormat::TXT => { "TXT" },
        }));
        match self.ftype {
            FileFormat::CSV=> {
                s.push_str(&format!("Has header: {}\n", self.header));
                s.push_str(&format!("Label index: {}\n", self.label_idx));
            },
            _ => { },
        }
        s.push_str(&format!("Delemeter: [{}]", self.delimeter));
        s
    }

    pub fn set_feature_size(&mut self, size: usize) {
        self.feature_size = size;
    }

    pub fn set_label_index(&mut self, idx: usize) {
        self.label_idx = idx;
    }

    pub fn set_delimeter(&mut self, delim: char) {
        self.delimeter = delim;
    }
}

fn count(mut hash_map: HashMap<char, u32>, word: char) -> HashMap<char, u32> {
    {
        let c = hash_map.entry(word).or_insert(0);
        *c += 1;
    }
    hash_map
}

pub fn infer(file_name: &str) -> InputFormat {
    let mut file = File::open(file_name.to_string()).unwrap();
    let mut reader = BufReader::new(file);

    // check CSV or TXT
    let mut first_line = String::new();
    reader.read_line(&mut first_line).unwrap();
    let mut input_format =  if first_line.contains(":") {
        InputFormat::txt_format()
    } else {
        InputFormat::csv_format()
    };

    // Check delimeter
    let reg = match input_format.ftype {
        FileFormat::CSV => {
            Regex::new(r"[+-]?\d+(,\d+)*(.\d+(e\d+)?)?").unwrap()
        },
        FileFormat::TXT => {
            Regex::new(r"\d+:[+-]?\d+(,\d+)*(.\d+(e\d+)?)?").unwrap()
        },
    };
    let mut second_line = String::new();
    reader.read_line(&mut second_line).expect("No second line to read");
    let caps = reg.captures(&second_line).unwrap().len();
    let second_line_after = reg.replace_all(&second_line, "");
    let cnt = second_line_after.chars()
                .fold(HashMap::new(), count);
    let default_delim: char = match input_format.ftype {
        FileFormat::CSV => { ',' },
        FileFormat::TXT => { '\t' },
    };
    let mut flag = false;
    if let Some(value) = cnt.get(&default_delim) {
        if value > &((caps as u32) - 2) {
            input_format.delimeter = default_delim;
            flag = true;
        }
    }
    if !flag {
        let mut max_cnt: u32 = 0;
        let mut delim = '\t';
        for (k, v) in &cnt {
            if v > &max_cnt {
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
                    input_format.header = if letter_caps.len() > 0 { true } else { false };
                },
                None => { },
            }
        },
        _ => { },
    };

    input_format
}


pub fn load_csv(file: &mut File, input_format: InputFormat) -> DataVec {
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut dv = Vec::new();

    let mut reader = BufReader::new(file);
    let mut l = String::new();
    if input_format.header {
        reader.read_line(&mut l).unwrap_or(0);
    }
    for line in reader.lines() {
        let content = line.unwrap();
        let mut v: Vec<ValueType> = content.split(input_format.delimeter)
                    .map(|x| x.parse::<ValueType>().unwrap())
                    .collect();
        dv.push(Data{
            label: v.swap_remove(input_format.label_idx),
            feature: v,
            target: 0.0,
            weight: 1.0,
            residual: 0.0,
            initial_guess: 0.0,
        })
    }
    dv
}

pub fn load_txt(file: &mut File, input_format: InputFormat) -> DataVec {
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut dv = Vec::new();

    let mut reader = BufReader::new(file);
    let mut label: ValueType = 0.0;
    let mut K: usize = 0;
    let mut V: ValueType = 0.0;
    //let mut v: Vec<ValueType> = Vec::with_capacity(input_format.feature_size);
    for line in reader.lines() {
        let mut v: Vec<ValueType> = vec![0.0; input_format.feature_size];
        for token in line.unwrap().split(input_format.delimeter) {
            let splited_token: Vec<&str> = token.split(":").collect();
            if splited_token.len() == 2 {
                let mut err = false;
                match splited_token[0 as usize].parse::<usize>() {
                    Ok(kk) => { K = kk; },
                    Err(_) => { err = true },
                }
                match splited_token[1 as usize].parse::<ValueType>() {
                    Ok(vv) => { V = vv; },
                    Err(_) => { err = true },
                }
                if K >= input_format.feature_size {
                    err = true;
                }
                if !err {
                    v[K] = V;
                }
            }
            if splited_token.len() == 1 {
                label = splited_token[0 as usize].parse::<ValueType>().unwrap();
            }
            else { // report error
            }
        }
        dv.push(Data{
            label: label,
            feature: v,
            target: 0.0,
            weight: 1.0,
            residual: 0.0,
            initial_guess: 0.0,
        });
    }

    dv
}

pub fn load(file_name: &str, input_format: InputFormat) -> DataVec {
    let mut file = File::open(file_name.to_string()).unwrap();
    match input_format.ftype {
        FileFormat::CSV => { load_csv(&mut file, input_format) }
        FileFormat::TXT => { load_txt(&mut file, input_format) }
    }
}