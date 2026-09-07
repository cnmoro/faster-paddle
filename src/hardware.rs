//! CPU budget discovery. Count only physical cores available to this process.

pub fn physical_cores() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or_else(|_| num_cpus::get())
        .max(1);
    #[cfg(target_os = "linux")]
    if let (Ok(status), Ok(cpuinfo)) = (
        std::fs::read_to_string("/proc/thread-self/status")
            .or_else(|_| std::fs::read_to_string("/proc/self/status")),
        std::fs::read_to_string("/proc/cpuinfo"),
    ) {
        if let Some(list) = status
            .lines()
            .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        {
            if let Some(count) = allowed_physical(&cpuinfo, list) {
                return count.min(logical).max(1);
            }
        }
    }
    num_cpus::get_physical().min(logical).max(1)
}

#[cfg(any(target_os = "linux", test))]
fn allowed_physical(cpuinfo: &str, allowed: &str) -> Option<usize> {
    let ranges: Option<Vec<(usize, usize)>> = allowed
        .trim()
        .split(',')
        .map(|v| {
            let (lo, hi) = v.split_once('-').unwrap_or((v, v));
            let (lo, hi) = (lo.parse().ok()?, hi.parse().ok()?);
            (lo <= hi).then_some((lo, hi))
        })
        .collect();
    let ranges = ranges?;
    let mut cores = std::collections::HashSet::new();
    for block in cpuinfo.split("\n\n") {
        let field = |key: &str| {
            block
                .lines()
                .filter_map(|l| l.split_once(':'))
                .find_map(|(k, v)| (k.trim() == key).then_some(v.trim()))
        };
        let Some(cpu) = field("processor").and_then(|v| v.parse::<usize>().ok()) else {
            continue;
        };
        if ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&cpu)) {
            // Missing topology (e.g. ARM) uses the platform fallback.
            cores.insert((field("physical id")?, field("core id")?));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affinity_counts_smt_siblings_once_and_handles_last_block() {
        let info = "processor: 0\nphysical id: 0\ncore id: 0\n\nprocessor: 1\nphysical id: 0\ncore id: 0\n\nprocessor: 2\nphysical id: 0\ncore id: 1";
        assert_eq!(allowed_physical(info, "0-2"), Some(2));
        assert_eq!(allowed_physical(info, "0-1"), Some(1));
        assert_eq!(allowed_physical(info, "1,2"), Some(2));
        assert_eq!(allowed_physical(info, "3-5"), None);
        assert_eq!(allowed_physical(info, "oops"), None);
        assert_eq!(allowed_physical("processor: 0", "0"), None);
    }
}
