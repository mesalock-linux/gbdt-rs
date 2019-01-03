//! This module implements some math functions used for gradient boosting process.

use crate::decision_tree::{DataVec, PredVec, ValueType};

/// Comparing two number with a costomized floating error threshold.
/// 
/// # Example
/// ```rust
/// use gbdt::fitness::almost_equal_thrs;
/// assert_eq!(true, almost_equal_thrs(1.0, 0.998, 0.01));
/// ```
pub fn almost_equal_thrs(a: ValueType, b: ValueType, thrs: f64) -> bool {
    ((a - b).abs() as f64) < thrs
}


/// Comparing two number with default floating error threshold.
/// 
/// # Example
/// ```rust
/// use gbdt::fitness::almost_equal;
/// assert_eq!(false, almost_equal(1.0, 0.998));
/// assert_eq!(true, almost_equal(1.0, 0.999998));
/// ```
pub fn almost_equal(a: ValueType, b: ValueType) -> bool {
    ((a - b).abs() as f64) < 1.0e-5
}

/// Return whether the first n data in data vector have same target values.
pub fn same(dv: &DataVec, len: usize) -> bool {
    assert!(dv.len() >= len);

    if len < 1 {
        return false;
    }

    let t: ValueType = dv[0].target;
    for i in dv.iter().skip(1) {
        if !(almost_equal(t, i.target)) {
            return false;
        }
    }
    true
}

/// Logistic value function.
pub fn logit(f: ValueType) -> ValueType {
    1.0 / (1.0 + (-2.0 * f).exp())
}

/// Negative binomial log-likelyhood loss function.
pub fn logit_loss(y: ValueType, f: ValueType) -> ValueType {
    2.0 * (1.0 + (-2.0 * y * f)).ln()
}

/// Log-likelyhood gradient calculation.
pub fn logit_loss_gradient(y: ValueType, f: ValueType) -> ValueType {
    2.0 * y / (1.0 + (2.0 * y * f).exp())
}

/// LAD loss function.
pub fn lad_loss(y: ValueType, f: ValueType) -> ValueType {
    (y - f).abs()
}

/// LAD gradient calculation.
pub fn lad_loss_gradient(y: ValueType, f: ValueType) -> ValueType {
    if y - f > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// RMSE (Root-Mean-Square deviation) calculation for first n element in data vector.
/// See [wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation) for detailed algorithm.
#[allow(non_snake_case)]
pub fn RMSE(dv: &DataVec, predict: &PredVec, len: usize) -> ValueType {
    assert_eq!(dv.len(), predict.len());
    assert!(dv.len() >= len);

    let mut s: ValueType = 0.0;
    let mut c: ValueType = 0.0;

    for i in 0..dv.len() {
        s += (predict[i] - dv[i].label).powf(2.0) * dv[i].weight;
        c += dv[i].weight;
    }
    s / c
}

/// MAE (Mean Absolute Error) calculation for first n element in data vector.
/// See [wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error) for detail for detailed algorithm.
#[allow(non_snake_case)]
pub fn MAE(dv: &DataVec, predict: &PredVec, len: usize) -> ValueType {
    assert_eq!(dv.len(), predict.len());
    assert!(dv.len() >= len);

    let mut s: ValueType = 0.0;
    let mut c: ValueType = 0.0;

    for i in 0..dv.len() {
        s += (predict[i] - dv[i].label).abs() * dv[i].weight;
        c += dv[i].weight;
    }
    s / c
}

/// AUC (Area Under the Curve) calculation for first n element in data vector.
/// See [wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) for detailed algorithm.
#[allow(non_snake_case)]
pub fn AUC(dv: &DataVec, predict: &PredVec, len: usize) -> ValueType {
    assert_eq!(dv.len(), predict.len());
    assert!(dv.len() >= len);

    let mut confusion_table: Vec<i32> = vec![0; 4];
    let threshold: ValueType = 0.5;
    let mut positive_scores: Vec<ValueType> = Vec::with_capacity(dv.len());
    let mut negative_scores: Vec<ValueType> = Vec::with_capacity(dv.len());

    // insert into confusion table
    for i in 0..dv.len() {
        if dv[i].label > 0.0 {
            positive_scores.push(predict[i]);
            if predict[i] >= threshold {
                confusion_table[3] += 1
            } else {
                confusion_table[2] += 1
            }
        } else {
            negative_scores.push(predict[i]);
            if predict[i] >= threshold {
                confusion_table[1] += 1
            } else {
                confusion_table[0] += 1
            }
        }
    }

    positive_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p_size = positive_scores.len();
    negative_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_size = positive_scores.len();
    if (p_size == 0) || (n_size == 0) {
        return 0.5;
    }

    let mut rank: ValueType = 1.0;
    let mut rank_sum: ValueType = 0.0;
    let mut pptr: usize = 0;
    let mut nptr: usize = 0;

    while pptr < p_size && nptr < n_size {
        let vp: ValueType = positive_scores[pptr];
        let vn: ValueType = negative_scores[nptr];
        if vn < vp {
            nptr += 1;
            rank += 1.0;
        } else if vp < vn {
            pptr += 1;
            rank += 1.0;
            rank_sum += rank;
        } else {
            let tie_score = vn;
            let mut kn: usize = 0;
            while nptr < n_size && almost_equal(negative_scores[nptr], tie_score) {
                kn += 1;
                nptr += 1;
            }
            let mut kp: usize = 0;
            while pptr < p_size && almost_equal(positive_scores[pptr], tie_score) {
                kp += 1;
                pptr += 1;
            }
            rank_sum += rank + ((kp + kn - 1) as ValueType) / 2.0;
            rank += (kp + kn) as ValueType;
        }
    }
    if pptr < p_size {
        rank_sum +=
            (rank + ((p_size - pptr - 1) as ValueType) / 2.0) * ((p_size - pptr) as ValueType);
        // TODO: double check if this is needed
        //rank += (p_size - pptr) as f64;
    }
    (rank_sum / (p_size as ValueType) - ((p_size as ValueType) + 1.0)) / (n_size as ValueType)
}


/// Return the weighted target average for first n data in data vector.
/// 
/// # Example
/// ```rust
/// use gbdt::decision_tree::{DataVec, Data, VALUE_TYPE_UNKNOWN};
/// use gbdt::fitness::{average, almost_equal};
/// let mut dv: DataVec = Vec::new();
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.1,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.2,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.3,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.4,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// assert!(almost_equal(0.3, average(&dv, dv.len())));
/// ```
pub fn average(dv: &DataVec, len: usize) -> ValueType {
    assert!(dv.len() >= len);

    if len == 0 {
        return 0.0;
    }

    let mut s: ValueType = 0.0;
    let mut c: ValueType = 0.0;
    for d in dv {
        s += d.weight * d.target;
        c += d.weight;
    }
    s / c
}

/// Return the weighted label average for first n data in data vector.
/// 
/// # Example
/// ```rust
/// use gbdt::decision_tree::{DataVec, Data, VALUE_TYPE_UNKNOWN};
/// use gbdt::fitness::{label_average, almost_equal};
/// let mut dv: DataVec = Vec::new();
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.1,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.2,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.3,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.4,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// assert!(almost_equal(0.4, label_average(&dv, dv.len())));
/// ```
pub fn label_average(dv: &DataVec, len: usize) -> ValueType {
    assert!(dv.len() >= len);
    let mut s: ValueType = 0.0;
    let mut c: ValueType = 0.0;
    for d in dv {
        s += d.label * d.weight;
        c += d.weight;
    }
    s / c
}

/// Return the weighted label median for first n data in data vector.
/// 
/// # Example
/// ```rust
/// use gbdt::decision_tree::{DataVec, Data, VALUE_TYPE_UNKNOWN};
/// use gbdt::fitness::{weighted_label_median, almost_equal};
/// let mut dv: DataVec = Vec::new();
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.1,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.2,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.3,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.4,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// assert!(almost_equal(0.0, weighted_label_median(&dv, dv.len())));
/// ```
pub fn weighted_label_median(dv: &DataVec, len: usize) -> ValueType {
    assert!(dv.len() >= len);
    let mut dv_copy = dv.to_vec();
    dv_copy.sort_by(|a, b| a.label.partial_cmp(&b.label).unwrap());
    let mut all_weight: ValueType = 0.0;
    for d in &dv_copy {
        all_weight += d.weight;
    }

    let mut weighted_median: ValueType = 0.0;
    let mut weight: ValueType = 0.0;

    for i in 0..len {
        weight += dv_copy[i].weight;
        if weight * 2.0 > all_weight {
            if i - 1 > 0 {
                weighted_median = (dv_copy[i].label + dv_copy[i - 1].label) / 2.0;
            } else {
                weighted_median = dv_copy[i].label;
            }
            break;
        }
    }
    weighted_median
}

/// Return the weighted residual median for first n data in data vector.
/// 
/// # Example
/// ```rust
/// use gbdt::decision_tree::{DataVec, Data, VALUE_TYPE_UNKNOWN};
/// use gbdt::fitness::{weighted_residual_median, almost_equal};
/// let mut dv: DataVec = Vec::new();
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.1,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 1.0,
///     weight: 0.2,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.3,
///     label: 1.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// dv.push(Data {
///     feature: Vec::new(),
///     target: 0.0,
///     weight: 0.4,
///     label: 0.0,
///     residual: 0.5,
///     initial_guess: VALUE_TYPE_UNKNOWN,
/// });
/// assert!(almost_equal(0.5, weighted_residual_median(&dv, dv.len())));
/// ```
pub fn weighted_residual_median(dv: &DataVec, len: usize) -> ValueType {
    assert!(dv.len() >= len);
    let mut dv_copy = dv.to_vec();
    dv_copy.sort_by(|a, b| a.residual.partial_cmp(&b.residual).unwrap());
    let mut all_weight: ValueType = 0.0;
    for d in &dv_copy {
        all_weight += d.weight;
    }

    let mut weighted_median: ValueType = 0.0;
    let mut weight: ValueType = 0.0;

    for i in 0..len {
        weight += dv_copy[i].weight;
        if weight * 2.0 > all_weight {
            if i - 1 > 0 {
                weighted_median = (dv_copy[i].residual + dv_copy[i - 1].residual) / 2.0;
            } else {
                weighted_median = dv_copy[i].residual;
            }
            break;
        }
    }
    weighted_median
}
