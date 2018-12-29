use decision_tree::{DataVec, PredVec, ValueType};

pub fn almost_equal(a: ValueType, b: ValueType) -> bool {
    (a - b).abs() < 1.0e-5
}

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

pub fn logit(f: ValueType) -> ValueType {
    1.0 / (1.0 + (-2.0 * f).exp())
}

pub fn logit_loss(y: ValueType, f: ValueType) -> ValueType {
    2.0 * (1.0 + (-2.0 * y * f)).ln()
}

pub fn logit_loss_gradient(y: ValueType, f: ValueType) -> ValueType {
    2.0 * y / (1.0 + (2.0 * y * f).exp())
}

pub fn lad_loss(y: ValueType, f: ValueType) -> ValueType {
    (y - f).abs()
}

pub fn lad_loss_gradient(y: ValueType, f: ValueType) -> ValueType {
    if y - f > 0.0 {
        1.0
    } else {
        -1.0
    }
}

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

#[allow(non_snake_case)]
pub fn AUC(dv: &DataVec, predict: &PredVec, len: usize) -> ValueType {
    assert_eq!(dv.len(), predict.len());
    assert!(dv.len() >= len);

    let mut confusion_table: Vec<i32> = vec![0; 4];
    let threshold: ValueType = 0.5;
    let mut positive_scores: Vec<ValueType> = Vec::new();
    let mut negative_scores: Vec<ValueType> = Vec::new();

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
            rank_sum += rank + ((kp + kn - 1) as f64) / 2.0;
            rank += (kp + kn) as f64;
        }
    }
    if pptr < p_size {
        rank_sum += (rank + ((p_size - pptr - 1) as f64) / 2.0) * ((p_size - pptr) as f64);
        // TODO: Double check if this is needed.
        //       Remove if possible.
        //rank += (p_size - pptr) as f64;
    }
    (rank_sum / (p_size as f64) - ((p_size as f64) + 1.0)) / (n_size as f64)
}

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
