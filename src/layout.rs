//! Port of util.py:extract_text_and_bounds — reconstructs reading-order text
//! with dynamic column/line detection, plus a bounds map.

use crate::ocr::OcrResult;
use serde::Serialize;

#[derive(Serialize)]
pub struct Bound {
    #[serde(rename = "topLeftCoord")]
    pub top_left: [i32; 2],
    #[serde(rename = "bottomRightCoord")]
    pub bottom_right: [i32; 2],
    pub text: String,
    pub confidence: f32,
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn stdev(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt()
}

/// Returns (full_text, bounds map keyed by original index).
pub fn extract_text_and_bounds(results: &[OcrResult]) -> (String, Vec<(usize, Bound)>) {
    let bounds: Vec<(usize, Bound)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            (
                i,
                Bound {
                    top_left: [r.box4[0], r.box4[1]],
                    bottom_right: [r.box4[2], r.box4[3]],
                    text: r.text.clone(),
                    confidence: r.score,
                },
            )
        })
        .collect();

    if results.is_empty() {
        return (String::new(), bounds);
    }

    // boxes as (x1,y1,x2,y2)
    let boxes: Vec<(f64, f64, f64, f64)> = results
        .iter()
        .map(|r| (r.box4[0] as f64, r.box4[1] as f64, r.box4[2] as f64, r.box4[3] as f64))
        .collect();

    // dynamic thresholds
    let mut heights: Vec<f64> = boxes.iter().map(|b| b.3 - b.1).filter(|&h| h > 0.0).collect();
    let widths: Vec<f64> = boxes.iter().map(|b| b.2 - b.0).filter(|&w| w > 0.0).collect();
    let y_threshold = if heights.is_empty() { 20.0 } else { median(&mut heights) * 0.8 };
    let x_gap_threshold = if widths.is_empty() { 50.0 } else { mean(&widths) * 0.5 };

    // centers
    let centers: Vec<f64> = boxes.iter().map(|b| (b.0 + b.2) / 2.0).collect();

    // column detection
    let mut unique_centers: Vec<f64> = centers.clone();
    unique_centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_centers.dedup();

    let column_assignments: Vec<Vec<usize>>;
    if unique_centers.len() <= 1 {
        column_assignments = vec![(0..boxes.len()).collect()];
    } else {
        let diffs: Vec<f64> = (0..unique_centers.len() - 1)
            .map(|i| unique_centers[i + 1] - unique_centers[i])
            .collect();
        let gap_threshold = if diffs.is_empty() {
            100.0
        } else {
            mean(&diffs) + 2.0 * stdev(&diffs)
        };
        // group unique centers into columns
        let mut columns: Vec<Vec<f64>> = Vec::new();
        let mut current = vec![unique_centers[0]];
        for i in 0..diffs.len() {
            if diffs[i] > gap_threshold {
                columns.push(current.clone());
                current = vec![unique_centers[i + 1]];
            } else {
                current.push(unique_centers[i + 1]);
            }
        }
        columns.push(current);
        let col_boundaries: Vec<f64> = columns.iter().map(|c| mean(c)).collect();
        let assignment_boundaries: Vec<f64> = (1..col_boundaries.len())
            .map(|i| (col_boundaries[i - 1] + col_boundaries[i]) / 2.0)
            .collect();
        let mut assigns: Vec<Vec<usize>> = vec![Vec::new(); col_boundaries.len()];
        for (j, &center) in centers.iter().enumerate() {
            let mut assigned = 0usize;
            for (k, &boundary) in assignment_boundaries.iter().enumerate() {
                if center > boundary {
                    assigned = k + 1;
                } else {
                    break;
                }
            }
            assigns[assigned].push(j);
        }
        column_assignments = assigns;
    }

    // items per column: (y1, x1, text, box)
    struct Item {
        y1: f64,
        x1: f64,
        text: String,
        x2: f64,
    }
    let mut column_items: Vec<Vec<Item>> = Vec::new();
    for indices in &column_assignments {
        if indices.is_empty() {
            continue;
        }
        let mut items: Vec<Item> = indices
            .iter()
            .map(|&j| Item {
                y1: boxes[j].1,
                x1: boxes[j].0,
                text: results[j].text.clone(),
                x2: boxes[j].2,
            })
            .collect();
        items.sort_by(|a, b| a.y1.partial_cmp(&b.y1).unwrap().then(a.x1.partial_cmp(&b.x1).unwrap()));
        column_items.push(items);
    }

    // sort columns left to right by avg x1
    column_items.sort_by(|a, b| {
        let ma = mean(&a.iter().map(|i| i.x1).collect::<Vec<_>>());
        let mb = mean(&b.iter().map(|i| i.x1).collect::<Vec<_>>());
        ma.partial_cmp(&mb).unwrap()
    });

    let mut full_text_parts: Vec<String> = Vec::new();
    for items in &column_items {
        let mut current_line: Vec<String> = Vec::new();
        let mut prev_y: Option<f64> = None;
        let mut prev_x_end: Option<f64> = None;
        for it in items {
            let ts = it.text.trim();
            if ts.is_empty() {
                continue;
            }
            let box_width = it.x2 - it.x1;
            if prev_y.is_none() || (it.y1 - prev_y.unwrap()).abs() > y_threshold {
                if !current_line.is_empty() {
                    full_text_parts.push(current_line.join(" "));
                }
                current_line = vec![ts.to_string()];
                prev_x_end = Some(it.x1 + box_width);
            } else {
                let current_x_start = it.x1;
                if prev_x_end.is_some() && current_x_start - prev_x_end.unwrap() < x_gap_threshold {
                    current_line.push(ts.to_string());
                    prev_x_end = Some(prev_x_end.unwrap().max(it.x1 + box_width));
                } else {
                    if !current_line.is_empty() {
                        full_text_parts.push(current_line.join(" "));
                    }
                    current_line = vec![ts.to_string()];
                    prev_x_end = Some(it.x1 + box_width);
                }
            }
            prev_y = Some(it.y1);
        }
        if !current_line.is_empty() {
            full_text_parts.push(current_line.join(" "));
        }
        full_text_parts.push(String::new());
    }

    // clean excessive empty lines
    let joined = full_text_parts.join("\n");
    let mut cleaned: Vec<String> = Vec::new();
    let mut prev_empty = false;
    for line in joined.split('\n') {
        if !line.trim().is_empty() {
            cleaned.push(line.to_string());
            prev_empty = false;
        } else if !prev_empty {
            cleaned.push(line.to_string());
            prev_empty = true;
        }
    }
    let full_text = cleaned.join("\n").trim().to_string();
    (full_text, bounds)
}
