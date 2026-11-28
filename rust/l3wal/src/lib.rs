//! GIL-free native L3 write-back seam (`L3-NATIVE-WRITEPATH-PROTOTYPE`).
//!
//! Promotes the proven PoC crate (`l3poc`) into the real seam. The native
//! worker owns the CPU/control hot path of one backup batch:
//!
//!   * chunked `pwritev` of the page blocks (directly from the source
//!     `(ptr,len)` pairs -> zero *intermediate gather* copy),
//!   * `FRAME_SET` + `FRAME_BATCH_COMMIT` payload/frame encoding + CRC,
//!   * WAL append + `fdatasync`,
//!   * page-lease bookkeeping (submit -> completion -> release).
//!
//! Durable ordering is preserved exactly: data `pwritev` -> WAL frames ->
//! `fdatasync`. Python only enqueues immutable offsets/lengths/metadata and
//! polls/receives completion; the worker never calls back into Python.
//!
//! Not production: this is the switchable A/B prototype. Arm A (the Python
//! reference path) is retained untouched.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// --------------------------------------------------------------------------- //
// CRC-32C (Castagnoli), reflected, poly 0x82F63B78 -- exact port of
// sglang/srt/mem_cache/storage/extent/extent_metadata.py:crc32c
// --------------------------------------------------------------------------- //

const CRC32C_POLY: u32 = 0x82F63B78;

const fn build_crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0usize;
    while index < 256 {
        let mut crc = index as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32C_POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
}

static CRC32C_TABLE: [u32; 256] = build_crc32c_table();

/// Python's `crc32c(data, crc=0)`, translated line for line.
fn crc32c_core(mut crc: u32, data: &[u8]) -> u32 {
    crc ^= u32::MAX;
    for &byte in data {
        crc = (crc >> 8) ^ CRC32C_TABLE[((crc ^ byte as u32) & 0xFF) as usize];
    }
    crc ^ u32::MAX
}

#[pyfunction]
#[pyo3(signature = (data, crc = 0))]
fn crc32c(data: &[u8], crc: u32) -> u32 {
    crc32c_core(crc, data)
}

/// `crc32c` over two disjoint slices without materialising the concatenation.
fn crc32c_pair(header: &[u8], payload: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in header {
        crc = (crc >> 8) ^ CRC32C_TABLE[((crc ^ byte as u32) & 0xFF) as usize];
    }
    for &byte in payload {
        crc = (crc >> 8) ^ CRC32C_TABLE[((crc ^ byte as u32) & 0xFF) as usize];
    }
    crc ^ u32::MAX
}

// --------------------------------------------------------------------------- //
// WAL frame / payload encoding -- exact ports of extent_metadata.py
// --------------------------------------------------------------------------- //

const FRAME_MAGIC: u32 = 0x484D3146;
const FORMAT_VERSION: u16 = 1;
const FRAME_HEADER_SIZE: usize = 26;
const DEFAULT_WAL_MAX_RECORD_BYTES: usize = 8 * 1024 * 1024;

const FRAME_SET: u8 = 1;
const FRAME_BATCH_COMMIT: u8 = 2;
const FRAME_GROUP_COMMIT: u8 = 3;
const FRAME_DROP_SLOT: u8 = 4;
const FRAME_CLEAR: u8 = 5;
const FRAME_CHECKPOINT_MARK: u8 = 6;

fn is_frame_type(frame_type: u8) -> bool {
    (FRAME_SET..=FRAME_CHECKPOINT_MARK).contains(&frame_type)
}

fn encode_frame_bytes(frame_type: u8, seq: u64, payload: &[u8]) -> Vec<u8> {
    const PACKED_HEADER_SIZE: usize = 22;
    let mut header = [0u8; PACKED_HEADER_SIZE];
    header[0..4].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[6] = frame_type;
    header[7] = 0; // flags
    header[8..10].copy_from_slice(&0u16.to_le_bytes()); // reserved
    header[10..14].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    header[14..22].copy_from_slice(&seq.to_le_bytes());

    let mut out = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    out.extend_from_slice(&header);
    // CRC covers the 22-byte packed header || payload; stored LE at offset 22.
    let checksum = crc32c_pair(&header, payload);
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

#[pyfunction]
#[pyo3(signature = (frame_type, seq, payload, wal_max_record_bytes = DEFAULT_WAL_MAX_RECORD_BYTES))]
fn encode_frame(
    py: Python<'_>,
    frame_type: u8,
    seq: u64,
    payload: &[u8],
    wal_max_record_bytes: usize,
) -> PyResult<Py<PyBytes>> {
    if !is_frame_type(frame_type) {
        return Err(PyValueError::new_err(format!(
            "unknown frame type: {frame_type}"
        )));
    }
    if payload.len() > wal_max_record_bytes {
        return Err(PyValueError::new_err(format!(
            "payload_len {} exceeds wal_max_record_bytes {}",
            payload.len(),
            wal_max_record_bytes
        )));
    }
    Ok(PyBytes::new(py, &encode_frame_bytes(frame_type, seq, payload)).unbind())
}

/// One index entry: `(storage_key, extent_id, offset, length, generation, component)`.
type EntryTuple = (Vec<u8>, u32, u32, u32, u64, u8);

fn encode_entry(
    out: &mut Vec<u8>,
    storage_key: &[u8],
    extent_id: u32,
    offset: u32,
    length: u32,
    generation: u64,
    component: u8,
) {
    out.extend_from_slice(&(storage_key.len() as u16).to_le_bytes());
    out.extend_from_slice(storage_key);
    out.extend_from_slice(&extent_id.to_le_bytes());
    out.extend_from_slice(&offset.to_le_bytes());
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&generation.to_le_bytes());
    out.push(component);
}

/// Exact port of `encode_set_payload(SetPayload(...))`.
fn encode_set_payload_bytes(
    group_epoch: &[u8; 16],
    group_seq: u64,
    batch_id: u64,
    member_index: u32,
    member_count: u32,
    entries: &[EntryTuple],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(40 + entries.len() * 32);
    out.extend_from_slice(group_epoch);
    out.extend_from_slice(&group_seq.to_le_bytes());
    out.extend_from_slice(&batch_id.to_le_bytes());
    out.extend_from_slice(&member_index.to_le_bytes());
    out.extend_from_slice(&member_count.to_le_bytes());
    for (key, extent_id, offset, length, generation, component) in entries {
        encode_entry(
            &mut out,
            key,
            *extent_id,
            *offset,
            *length,
            *generation,
            *component,
        );
    }
    out
}

/// Exact port of `encode_batch_commit_payload(BatchCommitPayload(...))`.
fn encode_batch_commit_payload_bytes(batch_id: u64, member_count: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&batch_id.to_le_bytes());
    out.extend_from_slice(&member_count.to_le_bytes());
    out
}

#[pyfunction]
fn encode_set_payload(
    py: Python<'_>,
    group_epoch: &[u8],
    group_seq: u64,
    batch_id: u64,
    member_index: u32,
    member_count: u32,
    entries: Vec<EntryTuple>,
) -> PyResult<Py<PyBytes>> {
    if group_epoch.len() != 16 {
        return Err(PyValueError::new_err(
            "group_epoch must be exactly 16 bytes",
        ));
    }
    let mut epoch = [0u8; 16];
    epoch.copy_from_slice(group_epoch);
    Ok(PyBytes::new(
        py,
        &encode_set_payload_bytes(
            &epoch,
            group_seq,
            batch_id,
            member_index,
            member_count,
            &entries,
        ),
    )
    .unbind())
}

#[pyfunction]
fn encode_batch_commit_payload(py: Python<'_>, batch_id: u64, member_count: u32) -> Py<PyBytes> {
    PyBytes::new(
        py,
        &encode_batch_commit_payload_bytes(batch_id, member_count),
    )
    .unbind()
}

// --------------------------------------------------------------------------- //
// Pointer-range validation (charter AMENDMENT 1-D)
// --------------------------------------------------------------------------- //

/// `true` iff `[ptr, ptr+len)` is fully contained in `[base, base+base_len)`.
fn range_within(ptr: u64, len: u64, base: u64, base_len: u64) -> bool {
    let end = match base.checked_add(base_len) {
        Some(end) => end,
        None => return false,
    };
    if ptr < base || ptr > end {
        return false;
    }
    match ptr.checked_add(len) {
        Some(stop) => stop <= end,
        None => false,
    }
}

#[pyfunction]
fn validate_range(ptr: u64, len: u64, base: u64, base_len: u64) -> bool {
    range_within(ptr, len, base, base_len)
}

// --------------------------------------------------------------------------- //
// `IOV_MAX` chunking (charter AMENDMENT 1-D)
// --------------------------------------------------------------------------- //

/// glibc `bits/confname.h`: `_SC_UIO_MAXIOV == 60`; this host returns 1024.
const SC_UIO_MAXIOV: libc::c_int = 60;
/// Linux kernel hard cap (`UIO_MAXIOV`), used only if `sysconf` is unavailable.
const UIO_MAXIOV_FALLBACK: u64 = 1024;

fn runtime_iov_max() -> u64 {
    let value = unsafe { libc::sysconf(SC_UIO_MAXIOV) };
    if value > 0 {
        value as u64
    } else {
        UIO_MAXIOV_FALLBACK
    }
}

#[pyfunction]
fn iov_max() -> u64 {
    runtime_iov_max()
}

#[pyfunction]
#[pyo3(signature = (count, iov_max = 0))]
fn plan_iovec_chunks(count: u64, iov_max: u64) -> PyResult<Vec<(u64, u64)>> {
    let hard = runtime_iov_max();
    let effective = if iov_max == 0 {
        hard
    } else {
        iov_max.min(hard)
    };
    if effective == 0 {
        return Err(PyValueError::new_err("effective IOV_MAX is zero"));
    }
    let mut plan = Vec::new();
    let mut start = 0u64;
    while start < count {
        let take = effective.min(count - start);
        plan.push((start, take));
        start += take;
    }
    Ok(plan)
}

// --------------------------------------------------------------------------- //
// Raw syscall helpers (all run on the worker thread, no GIL)
// --------------------------------------------------------------------------- //

/// `(errno, message)` for a failed native step. `errno == 0` means "no OS
/// error number" (e.g. a short write), so H2 callers can still distinguish it.
type IoError = (i32, String);

/// Last-OS-error as `(errno, "prefix: message")`.
fn os_error(prefix: String) -> IoError {
    let error = std::io::Error::last_os_error();
    let errno = error.raw_os_error().unwrap_or(0);
    (errno, format!("{prefix}: {error}"))
}

fn pairs_bytes(pairs: &[(u64, u64)]) -> u64 {
    pairs.iter().map(|(_, len)| *len).sum()
}

/// iovecs for the not-yet-written remainder of one chunk.
fn remaining_iovecs(chunk: &[(u64, u64)], already_written: u64) -> Vec<libc::iovec> {
    let mut skip = already_written;
    let mut iovecs: Vec<libc::iovec> = Vec::with_capacity(chunk.len());
    for &(ptr, len) in chunk {
        if skip >= len {
            skip -= len;
            continue;
        }
        iovecs.push(libc::iovec {
            iov_base: (ptr + skip) as *mut libc::c_void,
            iov_len: (len - skip) as usize,
        });
        skip = 0;
    }
    iovecs
}

/// Chunked, fully-drained `pwritev` at an explicit offset. Returns
/// `(bytes_written, syscall_count)`.
///
/// **H1:** every chunk is attempted in order and this only returns `Ok` when
/// *all* chunks were written in full. The first failing chunk aborts the entire
/// call, so the only caller (`run_job`) never enters WAL/commit on a partial
/// data write.
fn pwritev_pairs(fd: i32, offset: u64, pairs: &[(u64, u64)]) -> Result<(u64, u64), IoError> {
    let effective = runtime_iov_max();
    if effective == 0 {
        return Err((0, "effective IOV_MAX is zero".to_string()));
    }
    let mut cursor: libc::off_t = offset as libc::off_t;
    let mut total: u64 = 0;
    let mut syscalls: u64 = 0;
    let mut start = 0usize;
    while start < pairs.len() {
        let count = effective.min((pairs.len() - start) as u64) as usize;
        let chunk = &pairs[start..start + count];
        let chunk_bytes = pairs_bytes(chunk);
        let mut written: u64 = 0;
        while written < chunk_bytes {
            let iovecs = remaining_iovecs(chunk, written);
            if iovecs.is_empty() {
                break;
            }
            // SAFETY: iovecs point at caller-owned ranges that the caller keeps
            // alive until completion; iovcnt <= effective IOV_MAX by construction.
            let n =
                unsafe { libc::pwritev(fd, iovecs.as_ptr(), iovecs.len() as libc::c_int, cursor) };
            syscalls += 1;
            if n < 0 {
                return Err(os_error(format!("pwritev at offset {cursor} failed")));
            }
            if n == 0 {
                return Err((0, format!("pwritev wrote 0 bytes at offset {cursor}")));
            }
            let n = n as u64;
            written += n;
            total += n;
            cursor += n as libc::off_t;
        }
        start += count;
    }
    Ok((total, syscalls))
}

/// Fully-drained buffered `pwrite` at an explicit offset. Returns bytes written.
fn pwrite_all(fd: i32, buf: &[u8], offset: u64) -> Result<u64, IoError> {
    let mut written: u64 = 0;
    while (written as usize) < buf.len() {
        let n = unsafe {
            libc::pwrite(
                fd,
                buf[written as usize..].as_ptr() as *const libc::c_void,
                buf.len() - written as usize,
                (offset + written) as libc::off_t,
            )
        };
        if n < 0 {
            return Err(os_error(format!(
                "WAL pwrite at offset {} failed",
                offset + written
            )));
        }
        if n == 0 {
            return Err((
                0,
                format!("WAL pwrite wrote 0 bytes at offset {}", offset + written),
            ));
        }
        written += n as u64;
    }
    Ok(written)
}

fn fdatasync(fd: i32) -> Result<(), IoError> {
    let rc = unsafe { libc::fdatasync(fd) };
    if rc != 0 {
        return Err(os_error("fdatasync failed".to_string()));
    }
    Ok(())
}

// --------------------------------------------------------------------------- //
// Native write worker
// --------------------------------------------------------------------------- //

struct WriteJob {
    job_id: u64,
    data_fd: i32,
    /// One `(file_offset, [(ptr, len), ...])` group per extent.
    plans: Vec<(u64, Vec<(u64, u64)>)>,
    /// Python objects whose buffers were used (buffer path); kept alive until
    /// completion so their memory stays mapped.
    _keepalive: Vec<Py<PyAny>>,
    wal_fd: i32,
    wal_offset: u64,
    wal_seq: u64,
    group_epoch: [u8; 16],
    group_seq: u64,
    batch_id: u64,
    member_index: u32,
    member_count: u32,
    wal_max_record_bytes: u64,
    entries: Vec<EntryTuple>,
}

#[derive(Clone)]
struct Completion {
    job_id: u64,
    /// **H2:** explicit outcome. `false` means the batch MUST be treated as
    /// failed by Python; it is never "the worker merely finished".
    ok: bool,
    error: String,
    /// OS errno for the failing step (`0` on success or when no OS error
    /// number exists, e.g. a short write).
    errno: i32,
    bytes_written: u64,
    /// WAL byte offset/seq that Python may adopt. On a data-phase failure these
    /// equal the submitted values (no advance); after a fully written frame
    /// they advance by that frame only.
    wal_offset: u64,
    wal_seq: u64,
    data_syscalls: u64,
}

struct LeaseState {
    completed: bool,
    released: bool,
}

#[derive(Default)]
struct WriterStats {
    submitted: u64,
    completed: u64,
    failed: u64,
    cancelled: u64,
    range_validation_refusals: u64,
    leases_established: u64,
    leases_released: u64,
    release_before_completion_refused: u64,
}

struct WriterState {
    jobs: VecDeque<WriteJob>,
    completions: VecDeque<Completion>,
    results: HashMap<u64, Completion>,
    leases: HashMap<u64, LeaseState>,
    leased_ranges: HashMap<u64, Vec<(u64, u64)>>,
    /// Job currently owned by the worker thread (outside the queue).
    running: Option<u64>,
    stop: bool,
    /// Test rendezvous (H3): when armed, the worker parks after taking the next
    /// job and before touching any page bytes.
    park_armed: bool,
    parked_job: Option<u64>,
    resume: bool,
    next_job: u64,
    stats: WriterStats,
}

struct WriterInner {
    state: Mutex<WriterState>,
    work_ready: Condvar,
    job_done: Condvar,
    parked_signal: Condvar,
    resume_signal: Condvar,
}

fn run_job(job: &WriteJob) -> Completion {
    let mut bytes_written: u64 = 0;
    let mut data_syscalls: u64 = 0;
    for (offset, pairs) in &job.plans {
        match pwritev_pairs(job.data_fd, *offset, pairs) {
            Ok((written, syscalls)) => {
                bytes_written += written;
                data_syscalls += syscalls;
            }
            Err(error) => {
                return Completion {
                    job_id: job.job_id,
                    ok: false,
                    error: format!("data {}", error.1),
                    errno: error.0,
                    bytes_written,
                    wal_offset: job.wal_offset,
                    wal_seq: job.wal_seq,
                    data_syscalls,
                };
            }
        }
    }

    let mut wal_offset = job.wal_offset;
    let mut wal_seq = job.wal_seq;

    let set_payload = encode_set_payload_bytes(
        &job.group_epoch,
        job.group_seq,
        job.batch_id,
        job.member_index,
        job.member_count,
        &job.entries,
    );
    if set_payload.len() as u64 > job.wal_max_record_bytes {
        return Completion {
            job_id: job.job_id,
            ok: false,
            error: format!(
                "SET payload {} exceeds wal_max_record_bytes {}",
                set_payload.len(),
                job.wal_max_record_bytes
            ),
            errno: 0,
            bytes_written,
            wal_offset,
            wal_seq,
            data_syscalls,
        };
    }
    let frame_set = encode_frame_bytes(FRAME_SET, wal_seq, &set_payload);
    match pwrite_all(job.wal_fd, &frame_set, wal_offset) {
        Ok(n) => {
            wal_offset += n;
            wal_seq += 1;
        }
        Err(error) => {
            return Completion {
                job_id: job.job_id,
                ok: false,
                error: error.1,
                errno: error.0,
                bytes_written,
                wal_offset,
                wal_seq,
                data_syscalls,
            };
        }
    }

    // Batch commit member count is 1 in the frozen reference path.
    let commit_payload = encode_batch_commit_payload_bytes(job.batch_id, 1);
    let frame_commit = encode_frame_bytes(FRAME_BATCH_COMMIT, wal_seq, &commit_payload);
    match pwrite_all(job.wal_fd, &frame_commit, wal_offset) {
        Ok(n) => {
            wal_offset += n;
            wal_seq += 1;
        }
        Err(error) => {
            return Completion {
                job_id: job.job_id,
                ok: false,
                error: error.1,
                errno: error.0,
                bytes_written,
                wal_offset,
                wal_seq,
                data_syscalls,
            };
        }
    }

    if let Err(error) = fdatasync(job.wal_fd) {
        return Completion {
            job_id: job.job_id,
            ok: false,
            error: error.1,
            errno: error.0,
            bytes_written,
            wal_offset,
            wal_seq,
            data_syscalls,
        };
    }

    Completion {
        job_id: job.job_id,
        ok: true,
        error: String::new(),
        errno: 0,
        bytes_written,
        wal_offset,
        wal_seq,
        data_syscalls,
    }
}

/// Mark a queued/running job as completed with `completion`, updating the lease
/// and stats exactly once. **H2:** every enqueued request reaches here exactly
/// once (real completion or cancellation), so no request can be silently lost.
fn publish_completion(state: &mut WriterState, completion: Completion) {
    let job_id = completion.job_id;
    if let Some(lease) = state.leases.get_mut(&job_id) {
        lease.completed = true;
    }
    if completion.ok {
        state.stats.completed += 1;
    } else {
        state.stats.failed += 1;
    }
    state.completions.push_back(completion.clone());
    state.results.insert(job_id, completion);
}

/// **H3:** cancel every job still waiting in the queue. A running job is not
/// interruptible mid-syscall; it publishes its own real completion and its lease
/// is resolved then. Returns the number cancelled.
fn cancel_queued(state: &mut WriterState) -> u64 {
    let mut cancelled = 0;
    while let Some(job) = state.jobs.pop_front() {
        state.stats.cancelled += 1;
        cancelled += 1;
        publish_completion(
            state,
            Completion {
                job_id: job.job_id,
                ok: false,
                error: "cancelled before the worker started this job".to_string(),
                errno: 0,
                bytes_written: 0,
                wal_offset: job.wal_offset,
                wal_seq: job.wal_seq,
                data_syscalls: 0,
            },
        );
    }
    cancelled
}

fn worker_main(inner: Arc<WriterInner>) {
    loop {
        let job = {
            let mut state = inner.state.lock().unwrap();
            loop {
                if state.stop {
                    return;
                }
                if let Some(job) = state.jobs.pop_front() {
                    break job;
                }
                state = inner.work_ready.wait(state).unwrap();
            }
        };
        let job_id = job.job_id;
        // Optional test rendezvous (H3): park *before* any page byte is touched
        // so a shutdown/cancel test has a deterministic in-flight window.
        {
            let mut state = inner.state.lock().unwrap();
            state.running = Some(job_id);
            if state.park_armed {
                state.park_armed = false;
                state.parked_job = Some(job_id);
                inner.parked_signal.notify_all();
                while !state.resume && !state.stop {
                    state = inner.resume_signal.wait(state).unwrap();
                }
                state.resume = false;
                state.parked_job = None;
            }
        }
        let completion = run_job(&job);
        drop(job);
        {
            let mut state = inner.state.lock().unwrap();
            state.running = None;
            publish_completion(&mut state, completion);
            inner.job_done.notify_all();
        }
    }
}

/// `(job_id, ok, error, errno, bytes_written, wal_offset, wal_seq, data_syscalls)`.
type CompletionTuple = (u64, bool, String, i32, u64, u64, u64, u64);

fn completion_tuple(completion: Completion) -> CompletionTuple {
    (
        completion.job_id,
        completion.ok,
        completion.error,
        completion.errno,
        completion.bytes_written,
        completion.wal_offset,
        completion.wal_seq,
        completion.data_syscalls,
    )
}

#[pyclass]
struct Writer {
    inner: Arc<WriterInner>,
    join: Option<std::thread::JoinHandle<()>>,
    arenas: Mutex<Vec<(u64, u64)>>,
}

impl Writer {
    fn validate_sources(&self, plans: &[(u64, Vec<(u64, u64)>)]) -> PyResult<()> {
        let arenas = self.arenas.lock().unwrap().clone();
        if arenas.is_empty() {
            return Ok(());
        }
        for (_, pairs) in plans {
            for (ptr, len) in pairs {
                let inside = arenas
                    .iter()
                    .any(|(base, base_len)| range_within(*ptr, *len, *base, *base_len));
                if !inside {
                    self.inner.lock_stats_range_refusal();
                    return Err(PyValueError::new_err(format!(
                        "source ptr=0x{ptr:x} len={len} outside every registered pool \
                         arena; rejected before dereference"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl WriterInner {
    fn lock_stats_range_refusal(&self) {
        self.state.lock().unwrap().stats.range_validation_refusals += 1;
    }
}

#[pymethods]
impl Writer {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = Arc::new(WriterInner {
            state: Mutex::new(WriterState {
                jobs: VecDeque::new(),
                completions: VecDeque::new(),
                results: HashMap::new(),
                leases: HashMap::new(),
                leased_ranges: HashMap::new(),
                running: None,
                stop: false,
                park_armed: false,
                parked_job: None,
                resume: false,
                next_job: 0,
                stats: WriterStats::default(),
            }),
            work_ready: Condvar::new(),
            job_done: Condvar::new(),
            parked_signal: Condvar::new(),
            resume_signal: Condvar::new(),
        });
        let worker_inner = Arc::clone(&inner);
        let join = std::thread::Builder::new()
            .name("l3-native-wal-worker".to_string())
            .spawn(move || worker_main(worker_inner))
            .map_err(|err| PyValueError::new_err(format!("spawn failed: {err}")))?;
        Ok(Writer {
            inner,
            join: Some(join),
            arenas: Mutex::new(Vec::new()),
        })
    }

    /// Register the single pinned pool arena every raw `(ptr,len)` must lie in.
    fn register_arena(&self, base: u64, length: u64) {
        let mut arenas = self.arenas.lock().unwrap();
        if !arenas.iter().any(|entry| *entry == (base, length)) {
            arenas.push((base, length));
        }
    }

    fn arenas(&self) -> Vec<(u64, u64)> {
        self.arenas.lock().unwrap().clone()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data_fd, plans, wal_fd, wal_offset, wal_seq, group_epoch, group_seq,
                        batch_id, member_index, member_count, entries, wal_max_record_bytes))]
    fn submit_write_raw(
        &self,
        data_fd: i32,
        plans: Vec<(u64, Vec<(u64, u64)>)>,
        wal_fd: i32,
        wal_offset: u64,
        wal_seq: u64,
        group_epoch: Vec<u8>,
        group_seq: u64,
        batch_id: u64,
        member_index: u32,
        member_count: u32,
        entries: Vec<EntryTuple>,
        wal_max_record_bytes: u64,
    ) -> PyResult<u64> {
        self.validate_sources(&plans)?;
        self.enqueue(
            data_fd,
            plans,
            Vec::new(),
            wal_fd,
            wal_offset,
            wal_seq,
            group_epoch,
            group_seq,
            batch_id,
            member_index,
            member_count,
            entries,
            wal_max_record_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data_fd, plans, wal_fd, wal_offset, wal_seq, group_epoch, group_seq,
                        batch_id, member_index, member_count, entries, wal_max_record_bytes))]
    fn submit_write_buffers(
        &self,
        py: Python<'_>,
        data_fd: i32,
        plans: Vec<(u64, Vec<Py<PyAny>>)>,
        wal_fd: i32,
        wal_offset: u64,
        wal_seq: u64,
        group_epoch: Vec<u8>,
        group_seq: u64,
        batch_id: u64,
        member_index: u32,
        member_count: u32,
        entries: Vec<EntryTuple>,
        wal_max_record_bytes: u64,
    ) -> PyResult<u64> {
        let mut resolved: Vec<(u64, Vec<(u64, u64)>)> = Vec::with_capacity(plans.len());
        let mut keepalive: Vec<Py<PyAny>> = Vec::new();
        for (offset, sources) in plans {
            let mut pairs = Vec::with_capacity(sources.len());
            for source in sources {
                let (ptr, len) = {
                    let bound = source.bind(py);
                    let buffer = PyBuffer::<u8>::get(bound).map_err(|_| {
                        PyValueError::new_err("source does not export a contiguous u8 buffer")
                    })?;
                    if !buffer.is_c_contiguous() {
                        return Err(PyValueError::new_err("source buffer is not C-contiguous"));
                    }
                    (buffer.buf_ptr() as u64, buffer.len_bytes() as u64)
                };
                pairs.push((ptr, len));
                keepalive.push(source);
            }
            resolved.push((offset, pairs));
        }
        self.enqueue(
            data_fd,
            resolved,
            keepalive,
            wal_fd,
            wal_offset,
            wal_seq,
            group_epoch,
            group_seq,
            batch_id,
            member_index,
            member_count,
            entries,
            wal_max_record_bytes,
        )
    }

    /// Block until `job_id` completes (GIL released). Returns
    /// `(job_id, ok, error, errno, bytes_written, wal_offset, wal_seq, data_syscalls)`.
    #[pyo3(signature = (job_id, timeout_s = -1.0))]
    fn wait(
        &self,
        py: Python<'_>,
        job_id: u64,
        timeout_s: f64,
    ) -> PyResult<Option<CompletionTuple>> {
        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || {
            let deadline = if timeout_s < 0.0 {
                None
            } else {
                Some(Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0)))
            };
            let mut state = inner.state.lock().unwrap();
            loop {
                if let Some(completion) = state.results.get(&job_id) {
                    return Some(completion.clone());
                }
                let now = Instant::now();
                match deadline {
                    None => {
                        state = inner.job_done.wait(state).unwrap();
                    }
                    Some(deadline) => {
                        if now >= deadline {
                            return None;
                        }
                        let (guard, timeout) =
                            inner.job_done.wait_timeout(state, deadline - now).unwrap();
                        state = guard;
                        if timeout.timed_out() && !state.results.contains_key(&job_id) {
                            return None;
                        }
                    }
                }
            }
        });
        Ok(result.map(completion_tuple))
    }

    /// Drain completions without blocking.
    fn poll(&self) -> Vec<CompletionTuple> {
        let mut state = self.inner.state.lock().unwrap();
        state.completions.drain(..).map(completion_tuple).collect()
    }

    fn submitted(&self) -> u64 {
        self.inner.state.lock().unwrap().stats.submitted
    }

    fn completed(&self) -> u64 {
        self.inner.state.lock().unwrap().stats.completed
    }

    fn failed(&self) -> u64 {
        self.inner.state.lock().unwrap().stats.failed
    }

    fn pending(&self) -> usize {
        self.inner.state.lock().unwrap().jobs.len()
    }

    /// Jobs enqueued plus the job the worker currently owns. **H3:** teardown
    /// waits until this is zero.
    fn inflight(&self) -> usize {
        let state = self.inner.state.lock().unwrap();
        state.jobs.len() + usize::from(state.running.is_some())
    }

    /// Leases established but not yet released. Must be 0 before any pinned
    /// allocation is destroyed.
    fn outstanding_leases(&self) -> usize {
        let state = self.inner.state.lock().unwrap();
        state
            .leases
            .values()
            .filter(|lease| !lease.released)
            .count()
    }

    /// **H3:** wait (GIL released) until the queue and the running job are both
    /// empty. `true` if drained within `timeout_s` (`< 0` waits forever).
    #[pyo3(signature = (timeout_s = -1.0))]
    fn drain(&self, py: Python<'_>, timeout_s: f64) -> bool {
        let inner = Arc::clone(&self.inner);
        py.detach(move || {
            let deadline = if timeout_s < 0.0 {
                None
            } else {
                Some(Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0)))
            };
            let mut state = inner.state.lock().unwrap();
            loop {
                if state.jobs.is_empty() && state.running.is_none() {
                    return true;
                }
                let now = Instant::now();
                match deadline {
                    None => {
                        state = inner.job_done.wait(state).unwrap();
                    }
                    Some(deadline) => {
                        if now >= deadline {
                            return false;
                        }
                        let (guard, _timeout) =
                            inner.job_done.wait_timeout(state, deadline - now).unwrap();
                        state = guard;
                    }
                }
            }
        })
    }

    /// **H3:** cancel every queued-but-not-started job. Each cancelled job gets
    /// exactly one failure completion (H2: no silent drop) and its lease becomes
    /// releasable. A running job is not interrupted; it publishes its own real
    /// completion. Returns the number cancelled.
    fn cancel_pending(&self) -> u64 {
        let mut state = self.inner.state.lock().unwrap();
        let cancelled = cancel_queued(&mut state);
        self.inner.job_done.notify_all();
        cancelled
    }

    /// Test-only rendezvous (H3): park the worker after it takes the next job and
    /// before it touches any page byte.
    fn arm_test_rendezvous(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.park_armed = true;
    }

    /// Block until the worker has parked (H3 test rendezvous).
    fn wait_parked(&self, timeout_s: f64) -> bool {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0));
        let mut state = self.inner.state.lock().unwrap();
        while state.parked_job.is_none() {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (guard, timeout) = self
                .inner
                .parked_signal
                .wait_timeout(state, deadline - now)
                .unwrap();
            state = guard;
            if timeout.timed_out() && state.parked_job.is_none() {
                return false;
            }
        }
        true
    }

    /// Release a parked worker (H3 test rendezvous).
    fn resume(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.resume = true;
        self.inner.resume_signal.notify_all();
    }

    /// Invariant 1/2: is this job's page lease still active (unreleased)?
    fn lease_active(&self, job_id: u64) -> bool {
        let state = self.inner.state.lock().unwrap();
        state
            .leases
            .get(&job_id)
            .is_some_and(|lease| !lease.released)
    }

    /// Invariant 2: an attempt to reuse/overwrite `[ptr, ptr+len)` while a lease
    /// covers it must be refused. Returns the conflicting lease id, or 0.
    fn recompute_conflict(&self, ptr: u64, len: u64) -> u64 {
        let state = self.inner.state.lock().unwrap();
        for (lease_id, ranges) in &state.leased_ranges {
            let active = state
                .leases
                .get(lease_id)
                .is_some_and(|lease| !lease.released);
            if !active {
                continue;
            }
            for (base, length) in ranges {
                if ranges_overlap(ptr, len, *base, *length) {
                    return *lease_id;
                }
            }
        }
        0
    }

    /// Invariant 3: release a lease only after its completion arrived.
    fn release(&self, job_id: u64) -> PyResult<()> {
        let mut state = self.inner.state.lock().unwrap();
        let released = match state.leases.get_mut(&job_id) {
            Some(lease) => {
                if !lease.completed {
                    state.stats.release_before_completion_refused += 1;
                    return Err(PyValueError::new_err(format!(
                        "release refused: job {job_id} has no completion yet; \
                         commit_install must follow completion"
                    )));
                }
                lease.released = true;
                true
            }
            None => false,
        };
        if released {
            state.leased_ranges.remove(&job_id);
            state.stats.leases_released += 1;
        }
        Ok(())
    }

    /// **H3:** stop accepting work, let the running job finish (the worker is
    /// never killed mid-`pwritev`), join the thread, then resolve every queued
    /// job with a cancellation completion so no lease can leak. When this
    /// returns, no worker thread can touch a caller pointer any more.
    fn shutdown(&mut self) {
        {
            let mut state = self.inner.state.lock().unwrap();
            state.stop = true;
            state.resume = true;
        }
        self.inner.work_ready.notify_all();
        self.inner.job_done.notify_all();
        self.inner.resume_signal.notify_all();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        {
            let mut state = self.inner.state.lock().unwrap();
            cancel_queued(&mut state);
        }
        self.inner.job_done.notify_all();
    }

    /// Number of jobs cancelled by `cancel_pending`/`shutdown`.
    fn cancelled(&self) -> u64 {
        self.inner.state.lock().unwrap().stats.cancelled
    }

    fn stats(&self) -> Vec<(String, u64)> {
        let state = self.inner.state.lock().unwrap();
        let stats = &state.stats;
        vec![
            ("submitted".into(), stats.submitted),
            ("completed".into(), stats.completed),
            ("failed".into(), stats.failed),
            ("cancelled".into(), stats.cancelled),
            (
                "range_validation_refusals".into(),
                stats.range_validation_refusals,
            ),
            ("leases_established".into(), stats.leases_established),
            ("leases_released".into(), stats.leases_released),
            (
                "release_before_completion_refused".into(),
                stats.release_before_completion_refused,
            ),
        ]
    }
}

impl Writer {
    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &self,
        data_fd: i32,
        plans: Vec<(u64, Vec<(u64, u64)>)>,
        keepalive: Vec<Py<PyAny>>,
        wal_fd: i32,
        wal_offset: u64,
        wal_seq: u64,
        group_epoch: Vec<u8>,
        group_seq: u64,
        batch_id: u64,
        member_index: u32,
        member_count: u32,
        entries: Vec<EntryTuple>,
        wal_max_record_bytes: u64,
    ) -> PyResult<u64> {
        if group_epoch.len() != 16 {
            return Err(PyValueError::new_err(
                "group_epoch must be exactly 16 bytes",
            ));
        }
        let mut epoch = [0u8; 16];
        epoch.copy_from_slice(&group_epoch);
        let mut leased_ranges: Vec<(u64, u64)> = Vec::new();
        for (_, pairs) in &plans {
            leased_ranges.extend(pairs.iter().copied());
        }
        let mut state = self.inner.state.lock().unwrap();
        let job_id = state.next_job;
        state.next_job += 1;
        state.leases.insert(
            job_id,
            LeaseState {
                completed: false,
                released: false,
            },
        );
        state.leased_ranges.insert(job_id, leased_ranges.clone());
        state.stats.leases_established += 1;
        state.stats.submitted += 1;
        state.jobs.push_back(WriteJob {
            job_id,
            data_fd,
            plans,
            _keepalive: keepalive,
            wal_fd,
            wal_offset,
            wal_seq,
            group_epoch: epoch,
            group_seq,
            batch_id,
            member_index,
            member_count,
            wal_max_record_bytes,
            entries,
        });
        self.inner.work_ready.notify_one();
        Ok(job_id)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn ranges_overlap(a_ptr: u64, a_len: u64, b_ptr: u64, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    match (a_ptr.checked_add(a_len), b_ptr.checked_add(b_len)) {
        (Some(a_end), Some(b_end)) => a_ptr < b_end && b_ptr < a_end,
        _ => true,
    }
}

// --------------------------------------------------------------------------- //
// Standalone lease/generation registry -- carried over from the proven PoC so
// the completion-ordering invariants (and their negative control) stay directly
// testable against the same state machine the worker uses.
// --------------------------------------------------------------------------- //

#[derive(Default)]
struct LeaseStats {
    slots_allocated: u64,
    leases_established: u64,
    stale_generation_refused: u64,
    out_of_range_refused: u64,
    recycle_attempts: u64,
    recycle_refused_leased: u64,
    recycle_allowed: u64,
    release_before_completion_refused: u64,
    clean_releases: u64,
}

struct SlotRecord {
    generation: u64,
    ptr: u64,
    len: u64,
    free: bool,
}

struct RegistryLease {
    slot: u64,
    generation: u64,
    completed: bool,
}

#[derive(Default)]
struct RegistryState {
    next_slot: u64,
    next_generation: u64,
    next_lease: u64,
    slots: HashMap<u64, SlotRecord>,
    leases: HashMap<u64, RegistryLease>,
    stats: LeaseStats,
}

#[pyclass]
struct LeaseRegistry {
    arena_base: u64,
    arena_len: u64,
    state: Arc<Mutex<RegistryState>>,
}

#[pymethods]
impl LeaseRegistry {
    #[new]
    fn new(arena_base: u64, arena_len: u64) -> Self {
        LeaseRegistry {
            arena_base,
            arena_len,
            state: Arc::new(Mutex::new(RegistryState::default())),
        }
    }

    fn arena(&self) -> (u64, u64) {
        (self.arena_base, self.arena_len)
    }

    fn alloc_slot(&self, ptr: u64, len: u64) -> PyResult<(u64, u64)> {
        if !range_within(ptr, len, self.arena_base, self.arena_len) {
            let mut state = self.state.lock().unwrap();
            state.stats.out_of_range_refused += 1;
            return Err(PyValueError::new_err("alloc_slot: ptr outside arena"));
        }
        let mut state = self.state.lock().unwrap();
        state.next_slot += 1;
        state.next_generation += 1;
        let slot = state.next_slot;
        let generation = state.next_generation;
        state.slots.insert(
            slot,
            SlotRecord {
                generation,
                ptr,
                len,
                free: false,
            },
        );
        state.stats.slots_allocated += 1;
        Ok((slot, generation))
    }

    fn free_slot(&self, slot: u64) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        match state.slots.get_mut(&slot) {
            Some(record) => {
                record.free = true;
                Ok(())
            }
            None => Err(PyValueError::new_err("free_slot: unknown slot")),
        }
    }

    fn submit_lease(&self, slot: u64, generation: u64) -> PyResult<u64> {
        let mut state = self.state.lock().unwrap();
        let current = match state.slots.get(&slot) {
            Some(record) => record.generation,
            None => return Err(PyValueError::new_err("submit_lease: unknown slot")),
        };
        if current != generation {
            state.stats.stale_generation_refused += 1;
            return Err(PyValueError::new_err("submit_lease: stale generation"));
        }
        if state.leases.values().any(|lease| lease.slot == slot) {
            return Err(PyValueError::new_err("submit_lease: slot already leased"));
        }
        state.next_lease += 1;
        let lease_id = state.next_lease;
        state.leases.insert(
            lease_id,
            RegistryLease {
                slot,
                generation,
                completed: false,
            },
        );
        state.stats.leases_established += 1;
        Ok(lease_id)
    }

    fn mark_completed(&self, lease_id: u64) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        match state.leases.get_mut(&lease_id) {
            Some(lease) => {
                lease.completed = true;
                Ok(())
            }
            None => Err(PyValueError::new_err("mark_completed: unknown lease")),
        }
    }

    fn try_recycle(
        &self,
        slot: u64,
        new_generation: u64,
        src_ptr: u64,
        src_len: u64,
    ) -> PyResult<bool> {
        let mut state = self.state.lock().unwrap();
        state.stats.recycle_attempts += 1;
        let dst = match state.slots.get(&slot) {
            Some(record) => (record.ptr, record.len),
            None => return Err(PyValueError::new_err("try_recycle: unknown slot")),
        };
        if state.leases.values().any(|lease| lease.slot == slot) {
            state.stats.recycle_refused_leased += 1;
            return Ok(false);
        }
        let copy_len = src_len.min(dst.1) as usize;
        if copy_len > 0 {
            // SAFETY: caller contract from the PoC test harness; both ranges
            // live in the caller-registered arena.
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr as *const u8, dst.0 as *mut u8, copy_len);
            }
        }
        if let Some(record) = state.slots.get_mut(&slot) {
            record.generation = new_generation;
        }
        state.stats.recycle_allowed += 1;
        Ok(true)
    }

    fn commit_install_and_release(&self, lease_id: u64) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        let (slot, completed) = match state.leases.get(&lease_id) {
            Some(lease) => (lease.slot, lease.completed),
            None => {
                return Err(PyValueError::new_err(
                    "commit_install_and_release: unknown lease",
                ));
            }
        };
        if !completed {
            state.stats.release_before_completion_refused += 1;
            return Err(PyValueError::new_err("release refused: no completion yet"));
        }
        state.leases.remove(&lease_id);
        if let Some(record) = state.slots.get_mut(&slot) {
            record.free = true;
        }
        state.stats.clean_releases += 1;
        Ok(())
    }

    fn stats(&self) -> Vec<(String, u64)> {
        let state = self.state.lock().unwrap();
        let stats = &state.stats;
        vec![
            ("slots_allocated".into(), stats.slots_allocated),
            ("leases_established".into(), stats.leases_established),
            (
                "stale_generation_refused".into(),
                stats.stale_generation_refused,
            ),
            ("out_of_range_refused".into(), stats.out_of_range_refused),
            ("recycle_attempts".into(), stats.recycle_attempts),
            (
                "recycle_refused_leased".into(),
                stats.recycle_refused_leased,
            ),
            ("recycle_allowed".into(), stats.recycle_allowed),
            (
                "release_before_completion_refused".into(),
                stats.release_before_completion_refused,
            ),
            ("clean_releases".into(), stats.clean_releases),
        ]
    }
}

#[pymodule]
fn _l3wal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Writer>()?;
    module.add_class::<LeaseRegistry>()?;
    module.add_function(wrap_pyfunction!(crc32c, module)?)?;
    module.add_function(wrap_pyfunction!(encode_frame, module)?)?;
    module.add_function(wrap_pyfunction!(encode_set_payload, module)?)?;
    module.add_function(wrap_pyfunction!(encode_batch_commit_payload, module)?)?;
    module.add_function(wrap_pyfunction!(validate_range, module)?)?;
    module.add_function(wrap_pyfunction!(iov_max, module)?)?;
    module.add_function(wrap_pyfunction!(plan_iovec_chunks, module)?)?;
    module.add("FRAME_MAGIC", FRAME_MAGIC)?;
    module.add("FORMAT_VERSION", FORMAT_VERSION)?;
    module.add("FRAME_HEADER_SIZE", FRAME_HEADER_SIZE)?;
    module.add("FRAME_SET", FRAME_SET)?;
    module.add("FRAME_BATCH_COMMIT", FRAME_BATCH_COMMIT)?;
    module.add("DEFAULT_WAL_MAX_RECORD_BYTES", DEFAULT_WAL_MAX_RECORD_BYTES)?;
    Ok(())
}
