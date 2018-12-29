use rand::prelude::SliceRandom;
use rand::thread_rng;
use config::{Config, LOSS};
use decision_tree::DecisionTree;
use decision_tree::{DataVec, PredVec, ValueType, VALUE_TYPE_UNKNOWN};
use fitness::*;

pub struct GBDT {
    conf: Config,
    trees: Vec<DecisionTree>,
    bias: ValueType,
    pub gain: Vec<f64>,
}

impl GBDT {
    pub fn new(conf: &Config) -> GBDT {
        GBDT {
            conf: conf.clone(),
            trees: Vec::new(),
            bias: 0.0,
            gain: Vec::new(),
        }
    }

    pub fn init(&mut self, len: usize, dv: &DataVec) {
        assert!(dv.len() >= len);

        if self.conf.enable_initial_guess {
            return;
        }

        self.bias = match self.conf.loss {
            LOSS::SquaredError => label_average(dv, len),
            LOSS::LogLikelyhood => {
                let v: f64 = label_average(dv, len);
                ((1.0 + v) / (1.0 - v)).ln() / 2.0
            }
            LOSS::LAD => weighted_label_median(dv, len),
            LOSS::UnknownLoss => return,
        }
    }

    pub fn fit(&mut self, train_data: &DataVec) {
        self.trees = Vec::new();
        for _ in 0..self.conf.iterations {
            self.trees.push(DecisionTree::new());
        }
        let nr_samples: usize = if self.conf.data_sample_ratio < 1.0 {
            ((train_data.len() as f64) * self.conf.data_sample_ratio) as usize
        } else {
            train_data.len()
        };

        self.init(train_data.len(), &train_data);

        let mut train_data_copy = train_data.to_vec();

        let mut rng = thread_rng();
        for i in 0..self.conf.iterations {
            if nr_samples < train_data.len() {
                train_data_copy.shuffle(&mut rng);
            }
            if self.conf.loss == LOSS::SquaredError {
                self.square_loss_process(&mut train_data_copy, nr_samples, i);
            } else if self.conf.loss == LOSS::LogLikelyhood {
                self.log_loss_process(&mut train_data_copy, nr_samples, i);
            } else if self.conf.loss == LOSS::LAD {
                self.lad_loss_process(&mut train_data_copy, nr_samples, i);
            }
            self.trees[i].fit_n(&train_data_copy, nr_samples);
        }
    }

    pub fn predict_n(&self, test_data: &DataVec, iters: usize, n: usize) -> PredVec {
        assert!(iters <= self.conf.iterations);
        assert!(n <= test_data.len());

        if self.trees.is_empty() {
            return vec![VALUE_TYPE_UNKNOWN; test_data.len()];
        }

        let mut predicted: PredVec = Vec::new();
        for i in test_data.iter().take(n) {
            predicted.push(
                if self.conf.enable_initial_guess {
                    i.initial_guess
                } else {
                    self.bias
                }
            );
        }
        for i in 0..(iters) {
            let v: PredVec = self.trees[i].predict_n(test_data, n);
            for i in 0..v.len() {
                predicted[i] += self.conf.shrinkage * v[i];
            }
        }
        predicted.to_vec()
    }

    pub fn predict(&self, test_data: &DataVec, iters: usize) -> PredVec {
        self.predict_n(test_data, iters, test_data.len())
    }

    pub fn square_loss_process(&self, dv: &mut DataVec, samples: usize, iters: usize) {
        let predicted: PredVec = self.predict_n(&dv, iters, samples);
        for i in 0..samples {
            dv[i].target = dv[i].label - predicted[i];
        }
        if self.conf.debug {
            println!("RMSE = {}", RMSE(&dv, &predicted, samples));
        }
    }

    pub fn log_loss_process(&self, dv: &mut DataVec, samples: usize, iters: usize) {
        let predicted: PredVec = self.predict_n(&dv, iters, samples);
        for i in 0..samples {
            dv[i].target = logit_loss_gradient(dv[i].label, predicted[i]);
        }
    }

    pub fn lad_loss_process(&self, dv: &mut DataVec, samples: usize, iters: usize) {
        let predicted: PredVec = self.predict_n(&dv, iters, samples);
        for i in 0..samples {
            dv[i].residual = dv[i].label - predicted[i];
            dv[i].target = if dv[i].residual >= 0.0 { 1.0 } else { -1.0 };
        }
        if self.conf.debug {
            println!("MAE {}", MAE(&dv, &predicted, samples));
        }
    }
}
