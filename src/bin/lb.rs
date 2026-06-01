//! Load balancer: accepts client TCP connections on :9999 and hands each
//! accepted socket *file descriptor* to an API worker over a Unix socket
//! (`SCM_RIGHTS`). It never proxies bytes — after the hand-off the API talks to
//! the client directly — so the LB's per-request cost is one `sendmsg`.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("lb: Linux only (requires SCM_RIGHTS)");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
use std::env;
#[cfg(target_os = "linux")]
use std::os::fd::RawFd;
#[cfg(target_os = "linux")]
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::thread;
#[cfg(target_os = "linux")]
use std::time::Duration;

#[cfg(target_os = "linux")]
use rinha::net;

#[cfg(target_os = "linux")]
fn main() {
    let port: u16 = env_or("LB_PORT", 9999);
    let backlog: i32 = env_or("LB_BACKLOG", 65535);
    let accept_batch: usize = env_or("LB_ACCEPT_BATCH", 64);
    let sockets: Vec<PathBuf> = env::var("API_SOCKETS")
        .unwrap_or_else(|_| "/sockets/api1.sock,/sockets/api2.sock".to_string())
        .split(',')
        .map(|s| PathBuf::from(s.trim()))
        .collect();

    let listener = net::tcp_listener_opts(port, backlog, true).expect("bind :9999");
    net::set_nonblocking(listener).ok();
    net::set_tcp_defer_accept(listener, 1);

    // Connect a persistent control channel to each API (retry until they're up).
    let mut channels = Vec::with_capacity(sockets.len());
    for s in &sockets {
        loop {
            match net::uds_seqpacket_connect(s) {
                Ok(fd) => {
                    eprintln!("[lb] connected to {}", s.display());
                    channels.push(fd);
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(100)),
            }
        }
    }

    eprintln!(
        "[lb] listening on :{port}, {} backends, batch={accept_batch}, fd-passing mode",
        channels.len(),
    );

    // Round-robin hand-off. The only work per connection is a single fd send,
    // then we drop our copy.
    let n = channels.len();
    let mut rr = 0usize;
    loop {
        let mut accepted = 0usize;
        while accepted < accept_batch {
            let client = match net::accept_nb(listener) {
                net::Io::Ok(fd) => fd as RawFd,
                net::Io::WouldBlock => break,
                net::Io::Err(_) | net::Io::Eof => break,
            };
            accepted += 1;
            net::set_tcp_nodelay(client);
            net::set_quickack(client);

            let first = rr;
            rr = (rr + 1) % n;
            let mut sent = false;
            for attempt in 0..n {
                let ch = channels[(first + attempt) % n];
                if net::send_fd_nonblocking(ch, client).is_ok() {
                    sent = true;
                    break;
                }
            }
            if !sent {
                let _ = net::send_fd(channels[first], client);
            }
            // The API owns the fd now (or we failed to hand it off); close our copy.
            net::close_fd(client);
        }
        if accepted == 0 {
            wait_read(listener);
        }
    }
}

#[cfg(target_os = "linux")]
fn wait_read(fd: RawFd) {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe {
        libc::poll(&mut pfd, 1, -1);
    }
}

#[cfg(target_os = "linux")]
fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
