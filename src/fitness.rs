//! This module implements some math functions used for gradient boosting process.

use crate::decision_tree::{DataVec, PredVec, ValueType};

/// Comparing two number with a costomized floating error threshold.
///
/// # Example
/// ```rust
/// use gbdt::fitness::almost_equal_thrs;
/// assert_eq!(true, almost_equal_thrs(1.0, 0.998, 0.01));
/// ```
#[inline(always)]
pub fn almost_equal_thrs(a: ValueType, b: ValueType, thrs: f64) -> bool {
    f64::from((a - b).abs()) < thrs
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
    f64::from((a - b).abs()) < 1.0e-5
}

/// Return whether the first n data in data vector have same target values.
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
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
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
///
/// If the length of data vector and predicted vector is not same, it will panic.
#[allow(non_snake_case)]
pub fn RMSE(dv: &DataVec, predict: &PredVec, len: usize) -> ValueType {
    assert_eq!(dv.len(), predict.len());
    assert!(dv.len() >= len);

    let mut s: f64 = 0.0;
    let mut c: f64 = 0.0;

    for i in 0..dv.len() {
        s += (f64::from(predict[i]) - f64::from(dv[i].label)).powf(2.0) * f64::from(dv[i].weight);
        c += f64::from(dv[i].weight);
    }

    if c.abs() < 1e-10 {
        0.0
    } else {
        (s / c) as ValueType
    }
}

/// MAE (Mean Absolute Error) calculation for first n element in data vector.
/// See [wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error) for detail for detailed algorithm.
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
///
/// If the length of data vector and predicted vector is not same, it will panic.
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

struct AucPred {
    idx: usize,
    sorted_idx: ValueType,
    value: ValueType,
}

/// AUC (Area Under the Curve) calculation for first n element in data vector.
/// See [wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) for detailed algorithm.
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
///
/// If the length of data vector and predicted vector is not same, it will panic.
/// 
/// If the data vector contains only one class or more than two classes, it will panic.
#[allow(non_snake_case)]
pub fn AUC(dv: &DataVec, predict: &PredVec, len: usize) -> ValueType {
    assert_eq!(dv.len(), predict.len());
    assert!(dv.len() >= len);

    let mut classes: Vec<ValueType> = Vec::new();
    for i in 0..dv.len() {
        if !classes.contains(&dv[i].label) {
            classes.push(dv[i].label);
        }
    }
    assert!(classes.len() == 2);

    let mut preds: Vec<AucPred> = Vec::new();
    for i in 0..predict.len() {
        preds.push(AucPred {idx: i, sorted_idx: 0.0, value: predict[i]});
    }

    preds.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

    let mut ptr: usize = 0;
    let mut pptr: usize;

    while ptr < preds.len() {
        pptr = ptr;
        let mut sum = 0;
        while pptr < preds.len() && almost_equal(preds[pptr].value, preds[ptr].value) {
            sum += pptr;
            pptr += 1;
        }
        for i in ptr..pptr {
            preds[i].sorted_idx = sum as ValueType / (pptr - ptr) as ValueType;
        }
        ptr = pptr;
    }

    let mut cnt_pos: ValueType = 0.0;
    let mut ret: ValueType = 0.0;
    for i in 0..preds.len() {
        if dv[preds[i].idx].label == 1.0 {
            cnt_pos += 1.0;
            ret += preds[i].sorted_idx;
        }
    }

    ret = (ret - (cnt_pos-1.0)*cnt_pos/2.0) / (cnt_pos * (dv.len() as ValueType - cnt_pos));

    ret
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
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
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
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
pub fn label_average(dv: &DataVec, len: usize) -> ValueType {
    assert!(dv.len() >= len);
    let mut s: f64 = 0.0;
    let mut c: f64 = 0.0;
    for d in dv {
        s += f64::from(d.label) * f64::from(d.weight);
        c += f64::from(d.weight);
    }
    if c.abs() < 1e-10 {
        0.0
    } else {
        (s / c) as ValueType
    }
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
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
pub fn weighted_label_median(dv: &DataVec, len: usize) -> ValueType {
    assert!(dv.len() >= len);
    let mut dv_copy = dv.to_vec();
    dv_copy.sort_by(|a, b| a.label.partial_cmp(&b.label).unwrap());
    let mut all_weight: f64 = 0.0;
    for d in &dv_copy {
        all_weight += f64::from(d.weight);
    }

    let mut weighted_median: ValueType = 0.0;
    let mut weight: f64 = 0.0;

    for i in 0..len {
        weight += f64::from(dv_copy[i].weight);
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
///
/// # Panic
/// If the specified length is greater than the length of data vector, it will panic.
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
