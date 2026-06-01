//! Low-latency API server.
//!
//! In the official compose topology the Rust LB accepts TCP on :9999 and passes
//! each accepted client socket to an API instance over a Unix socket
//! (`SCM_RIGHTS`). The API then serves HTTP/1.1 keep-alive directly on that
//! client socket, so the LB stays outside the per-request hot path.
//!
//! If `API_SOCKET` is unset, this binary can still run in standalone benchmark
//! mode with SO_REUSEPORT workers.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("api: Linux only (requires epoll + SO_REUSEPORT)");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    linux::run();
}

#[cfg(target_os = "linux")]
mod linux {
    use std::os::fd::RawFd;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use memchr::memmem;
    use memmap2::Mmap;

    use rinha::index::IvfIndex;
    use rinha::net::{self, Epoll, Io, RecvFd};
    use rinha::{parser, responses, vectorizer};

    const MAX_EVENTS: usize = 1024;
    const READ_CHUNK: usize = 4096;
    const MAX_FD_SLOTS: usize = 65_536;
    const CONN_BUF_CAP: usize = 16 * 1024;
    const LISTEN_TOKEN: u64 = u64::MAX;

    struct Conn {
        buf: [u8; CONN_BUF_CAP],
        len: usize,
    }

    impl Conn {
        fn new_box() -> Box<Self> {
            Box::new(Conn {
                buf: [0; CONN_BUF_CAP],
                len: 0,
            })
        }

        #[inline]
        fn reset(&mut self) {
            self.len = 0;
        }
    }

    struct ConnTable {
        slots: Vec<Option<Box<Conn>>>,
        pool: Vec<Box<Conn>>,
        pool_cap: usize,
    }

    impl ConnTable {
        fn new(pool_cap: usize) -> Self {
            let mut pool = Vec::with_capacity(pool_cap);
            for _ in 0..pool_cap.min(128) {
                pool.push(Conn::new_box());
            }
            let mut slots = Vec::with_capacity(MAX_FD_SLOTS);
            slots.resize_with(MAX_FD_SLOTS, || None);
            ConnTable {
                slots,
                pool,
                pool_cap,
            }
        }

        fn insert(&mut self, fd: RawFd) -> bool {
            let idx = fd as usize;
            if idx >= self.slots.len() {
                return false;
            }
            let mut conn = self.pool.pop().unwrap_or_else(Conn::new_box);
            conn.reset();
            self.slots[idx] = Some(conn);
            true
        }

        #[inline]
        fn get_mut(&mut self, fd: RawFd) -> Option<&mut Conn> {
            self.slots.get_mut(fd as usize)?.as_deref_mut()
        }

        fn remove(&mut self, fd: RawFd) {
            let idx = fd as usize;
            if idx >= self.slots.len() {
                return;
            }
            if let Some(mut conn) = self.slots[idx].take() {
                conn.reset();
                if self.pool.len() < self.pool_cap {
                    self.pool.push(conn);
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    struct WaitTuning {
        spin: Duration,
        idle_us: u64,
    }

    pub fn run() {
        let port: u16 = env_or("PORT", 9999);
        let workers: usize = env_or("API_WORKERS", 2);
        let backlog: i32 = env_or("LISTEN_BACKLOG", 4096);
        let api_socket = std::env::var("API_SOCKET")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let index_path =
            std::env::var("INDEX_PATH").unwrap_or_else(|_| "/data/index.bin".to_string());
        let nprobe: usize = env_or("NPROBE", 10);

        // mmap + mlock the index so queries never page-fault. One mapping is
        // shared by all worker threads (read-only).
        let file = std::fs::File::open(&index_path)
            .unwrap_or_else(|e| panic!("open index {index_path}: {e}"));
        let mmap = unsafe { Mmap::map(&file).expect("mmap index") };
        net::mlock(mmap.as_ptr(), mmap.len());
        let mmap: &'static Mmap = Box::leak(Box::new(mmap));
        let data: &'static [u8] = &mmap[..];
        let index: &'static IvfIndex<'static> =
            Box::leak(Box::new(IvfIndex::from_bytes(data).expect("invalid index")));

        // Warm the query path (page-ins, branch predictor) before serving.
        warm_up(index, nprobe);

        if let Some(socket) = api_socket {
            eprintln!(
                "[api] index {} vectors, nprobe={nprobe}, socket={socket}",
                index.num_vectors
            );
            fd_worker(PathBuf::from(socket), backlog, index, nprobe);
            return;
        }

        eprintln!(
            "[api] index {} vectors, nprobe={nprobe}, standalone port={port}, workers={workers}",
            index.num_vectors
        );

        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let h = std::thread::Builder::new()
                .name(format!("worker-{w}"))
                .spawn(move || tcp_worker(w, port, backlog, index, nprobe))
                .expect("spawn worker");
            handles.push(h);
        }
        for h in handles {
            let _ = h.join();
        }
    }

    /// Official topology worker: receives client fds from the LB over a Unix
    /// socket, then owns those TCP sockets directly.
    fn fd_worker(socket_path: PathBuf, backlog: i32, index: &'static IvfIndex, nprobe: usize) {
        let listener = net::uds_seqpacket_listener(&socket_path, backlog)
            .unwrap_or_else(|e| panic!("[api] bind {}: {e}", socket_path.display()));
        net::set_nonblocking(listener).ok();

        let ep = Epoll::new().expect("epoll");
        configure_epoll(&ep);
        ep.add(listener, LISTEN_TOKEN)
            .expect("epoll add uds listener");

        let mut controls: Vec<RawFd> = Vec::with_capacity(4);
        let mut conns = ConnTable::new(env_or("CONN_POOL_CAP", 512usize));
        let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; MAX_EVENTS];
        let wait = wait_tuning();

        loop {
            let n = match wait_events(&ep, &mut events, wait) {
                Ok(n) => n,
                Err(_) => continue,
            };
            for ev in events.iter().take(n) {
                if ev.u64 == LISTEN_TOKEN {
                    loop {
                        match net::accept_nb(listener) {
                            Io::Ok(fd) => {
                                let fd = fd as RawFd;
                                if ep.add(fd, fd as u64).is_ok() {
                                    controls.push(fd);
                                } else {
                                    net::close_fd(fd);
                                }
                            }
                            _ => break,
                        }
                    }
                    continue;
                }

                let fd = ev.u64 as RawFd;
                if let Some(control_pos) = controls.iter().position(|&c| c == fd) {
                    if !drain_control(fd, &ep, &mut conns, index, nprobe) {
                        ep.del(fd);
                        controls.swap_remove(control_pos);
                        net::close_fd(fd);
                    }
                    continue;
                }

                let keep = match conns.get_mut(fd) {
                    Some(c) => handle_client(fd, c, index, nprobe),
                    None => false,
                };
                if !keep {
                    ep.del(fd);
                    net::close_fd(fd);
                    conns.remove(fd);
                }
            }
        }
    }

    fn drain_control(
        channel: RawFd,
        ep: &Epoll,
        conns: &mut ConnTable,
        index: &IvfIndex,
        nprobe: usize,
    ) -> bool {
        loop {
            match net::recv_fd(channel) {
                RecvFd::Fd(fd) => {
                    net::set_nonblocking(fd).ok();
                    if ep.add(fd, fd as u64).is_err() || !conns.insert(fd) {
                        net::close_fd(fd);
                        continue;
                    }
                    let keep = match conns.get_mut(fd) {
                        Some(c) => handle_client(fd, c, index, nprobe),
                        None => false,
                    };
                    if !keep {
                        ep.del(fd);
                        conns.remove(fd);
                        net::close_fd(fd);
                    }
                }
                RecvFd::WouldBlock => return true,
                RecvFd::Closed => return false,
            }
        }
    }

    /// Standalone benchmark worker: its own SO_REUSEPORT listener + epoll
    /// reactor. This mode is intentionally not used by the official compose.
    fn tcp_worker(w: usize, port: u16, backlog: i32, index: &'static IvfIndex, nprobe: usize) {
        let listener = net::tcp_listener_opts(port, backlog, true)
            .unwrap_or_else(|e| panic!("[api-w{w}] bind :{port}: {e}"));
        net::set_nonblocking(listener).ok();

        let ep = Epoll::new().expect("epoll");
        configure_epoll(&ep);
        ep.add(listener, LISTEN_TOKEN).expect("epoll add listener");

        let mut conns = ConnTable::new(env_or("CONN_POOL_CAP", 512usize));
        let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; MAX_EVENTS];
        let wait = wait_tuning();

        loop {
            let n = match wait_events(&ep, &mut events, wait) {
                Ok(n) => n,
                Err(_) => continue,
            };
            for ev in events.iter().take(n) {
                if ev.u64 == LISTEN_TOKEN {
                    // Drain the accept queue.
                    loop {
                        match net::accept_nb(listener) {
                            Io::Ok(cfd) => {
                                let cfd = cfd as RawFd;
                                net::set_tcp_nodelay(cfd);
                                net::set_quickack(cfd);
                                if ep.add(cfd, cfd as u64).is_err() || !conns.insert(cfd) {
                                    net::close_fd(cfd);
                                }
                            }
                            _ => break,
                        }
                    }
                    continue;
                }

                let fd = ev.u64 as RawFd;
                let keep = match conns.get_mut(fd) {
                    Some(c) => handle_client(fd, c, index, nprobe),
                    None => false,
                };
                if !keep {
                    ep.del(fd);
                    net::close_fd(fd);
                    conns.remove(fd);
                }
            }
        }
    }

    /// Drains readable bytes and answers every complete pipelined request.
    /// Returns false when the connection should be closed.
    fn handle_client(fd: RawFd, conn: &mut Conn, index: &IvfIndex, nprobe: usize) -> bool {
        loop {
            if conn.len == CONN_BUF_CAP {
                return false;
            }
            let cap = (CONN_BUF_CAP - conn.len).min(READ_CHUNK);
            match net::read(fd, &mut conn.buf[conn.len..conn.len + cap]) {
                Io::Ok(n) => {
                    conn.len += n;
                    if n < cap {
                        break;
                    }
                }
                Io::WouldBlock => break,
                Io::Eof | Io::Err(_) => return false,
            }
        }
        if conn.len == 0 {
            return true;
        }

        let mut start = 0usize;
        while start < conn.len {
            match parse_request(&conn.buf[start..conn.len]) {
                ParseResult::Need => break,
                ParseResult::Ready(consumed) => {
                    if net::write_all(fd, responses::READY_RESPONSE).is_err() {
                        return false;
                    }
                    start += consumed;
                }
                ParseResult::NotFound(consumed) => {
                    if net::write_all(fd, responses::NOTFOUND_RESPONSE).is_err() {
                        return false;
                    }
                    start += consumed;
                }
                ParseResult::Fraud {
                    consumed,
                    body_off,
                    body_len,
                } => {
                    let body = &conn.buf[start + body_off..start + body_off + body_len];
                    let fc = score(body, index, nprobe);
                    if net::write_all(fd, responses::full_response(fc)).is_err() {
                        return false;
                    }
                    start += consumed;
                }
            }
        }
        if start > 0 {
            if start == conn.len {
                conn.len = 0;
            } else {
                conn.buf.copy_within(start..conn.len, 0);
                conn.len -= start;
            }
        }
        true
    }

    #[inline]
    fn score(body: &[u8], index: &IvfIndex, nprobe: usize) -> usize {
        match parser::parse(body) {
            Some(p) => index.query(&vectorizer::vectorize(&p), nprobe).0 as usize,
            None => 0,
        }
    }

    fn warm_up(index: &IvfIndex, nprobe: usize) {
        let count: usize = env_or("API_WARMUP_QUERIES", 2048);
        let mut acc = 0u8;
        for i in 0..count {
            let mut v = [0.0f32; 16];
            for (d, slot) in v.iter_mut().enumerate().take(14) {
                *slot = (((i * 131 + d * 17) % 1000) as f32) / 1000.0;
            }
            if i & 3 == 0 {
                v[5] = -1.0;
                v[6] = -1.0;
            }
            acc ^= index.query(&v, nprobe).0;
        }
        std::hint::black_box(acc);
    }

    fn wait_tuning() -> WaitTuning {
        WaitTuning {
            spin: Duration::from_micros(env_or("EPOLL_SPIN_US", 30u64)),
            idle_us: env_or("EPOLL_IDLE_US", 80u64),
        }
    }

    fn configure_epoll(ep: &Epoll) {
        let usecs: u32 = env_or("EPOLL_BUSY_POLL_US", 0u32);
        let budget: u16 = env_or("EPOLL_BUSY_POLL_BUDGET", 8u16);
        let prefer: u32 = env_or("EPOLL_PREFER_BUSY_POLL", 0u32);
        ep.set_busy_poll(usecs, budget, prefer != 0);
    }

    fn wait_events(
        ep: &Epoll,
        events: &mut [libc::epoll_event],
        tuning: WaitTuning,
    ) -> std::io::Result<usize> {
        if tuning.spin.is_zero() {
            return if tuning.idle_us == 0 {
                ep.wait(events, -1)
            } else {
                ep.wait_micros(events, tuning.idle_us)
            };
        }

        let mut n = ep.wait(events, 0)?;
        if n != 0 {
            return Ok(n);
        }
        let start = Instant::now();
        while start.elapsed() < tuning.spin {
            n = ep.wait(events, 0)?;
            if n != 0 {
                return Ok(n);
            }
            std::hint::spin_loop();
        }
        if tuning.idle_us == 0 {
            ep.wait(events, -1)
        } else {
            ep.wait_micros(events, tuning.idle_us)
        }
    }

    enum ParseResult {
        Need,
        Ready(usize),
        NotFound(usize),
        Fraud {
            consumed: usize,
            body_off: usize,
            body_len: usize,
        },
    }

    /// Parses one HTTP request from the front of `buf`.
    fn parse_request(buf: &[u8]) -> ParseResult {
        let hdr_end = match memmem::find(buf, b"\r\n\r\n") {
            Some(p) => p + 4,
            None => return ParseResult::Need,
        };
        match buf[0] {
            b'P' => {
                let cl = content_length(&buf[..hdr_end]).unwrap_or(0);
                let total = hdr_end + cl;
                if buf.len() < total {
                    return ParseResult::Need;
                }
                ParseResult::Fraud {
                    consumed: total,
                    body_off: hdr_end,
                    body_len: cl,
                }
            }
            b'G' => {
                if request_path_is(buf, hdr_end, b"/ready") {
                    ParseResult::Ready(hdr_end)
                } else {
                    ParseResult::NotFound(hdr_end)
                }
            }
            _ => ParseResult::NotFound(hdr_end),
        }
    }

    /// Checks the request-line path (between the two spaces) equals `want`.
    fn request_path_is(buf: &[u8], hdr_end: usize, want: &[u8]) -> bool {
        let line = &buf[..hdr_end];
        let p1 = match memchr::memchr(b' ', line) {
            Some(p) => p + 1,
            None => return false,
        };
        let rest = &line[p1..];
        let p2 = match memchr::memchr(b' ', rest) {
            Some(p) => p,
            None => return false,
        };
        &rest[..p2] == want
    }

    /// Parses the Content-Length header value (case-insensitive key).
    fn content_length(headers: &[u8]) -> Option<usize> {
        let (pos, len) = if let Some(pos) = memmem::find(headers, b"Content-Length:") {
            (pos, b"Content-Length:".len())
        } else if let Some(pos) = memmem::find(headers, b"content-length:") {
            (pos, b"content-length:".len())
        } else {
            let pos = find_ci(headers, b"content-length:")?;
            (pos, b"content-length:".len())
        };
        let mut i = pos + len;
        while i < headers.len() && (headers[i] == b' ' || headers[i] == b'\t') {
            i += 1;
        }
        let mut v = 0usize;
        let mut saw = false;
        while i < headers.len() && headers[i].is_ascii_digit() {
            v = v * 10 + (headers[i] - b'0') as usize;
            i += 1;
            saw = true;
        }
        if saw {
            Some(v)
        } else {
            None
        }
    }

    /// Case-insensitive substring search (small needle, small haystack).
    fn find_ci(hay: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || hay.len() < needle.len() {
            return None;
        }
        let end = hay.len() - needle.len();
        let mut i = 0;
        while i <= end {
            let mut j = 0;
            while j < needle.len() && hay[i + j].to_ascii_lowercase() == needle[j] {
                j += 1;
            }
            if j == needle.len() {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
        std::env::var(key)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }
}
