//! Thin, low-level Linux networking helpers built directly on `libc`:
//! TCP/Unix listeners, `SCM_RIGHTS` file-descriptor passing, epoll, and the
//! socket tuning that matters for tail latency. Linux-only.

use std::ffi::c_void;
use std::io;
use std::mem;
use std::os::fd::RawFd;
use std::path::Path;

/// Result of a non-blocking read/recv attempt.
pub enum Io {
    Ok(usize),
    WouldBlock,
    Eof,
    Err(io::Error),
}

#[inline]
fn last_err() -> io::Error {
    io::Error::last_os_error()
}

#[inline]
pub fn close_fd(fd: RawFd) {
    unsafe {
        libc::close(fd);
    }
}

#[inline]
pub fn set_nonblocking(fd: RawFd) -> io::Result<()> {
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL, 0);
        if flags < 0 {
            return Err(last_err());
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(last_err());
        }
    }
    Ok(())
}

#[inline]
pub fn set_tcp_nodelay(fd: RawFd) {
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_NODELAY,
            &one as *const _ as *const c_void,
            mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

/// Enables TCP_QUICKACK (suppresses delayed-ACK stalls). Must be re-armed
/// periodically by the kernel, but setting it after accept helps the first
/// exchanges.
#[inline]
pub fn set_quickack(fd: RawFd) {
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_QUICKACK,
            &one as *const _ as *const c_void,
            mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

#[inline]
fn set_sockopt_flag(fd: RawFd, level: libc::c_int, name: libc::c_int) {
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            fd,
            level,
            name,
            &one as *const _ as *const c_void,
            mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

/// Creates a listening TCP socket on `0.0.0.0:port`. When `reuseport` is set,
/// SO_REUSEPORT lets several independent sockets share the same port and the
/// kernel load-balances incoming connections across them (one per worker).
pub fn tcp_listener_opts(port: u16, backlog: i32, reuseport: bool) -> io::Result<RawFd> {
    unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
        if fd < 0 {
            return Err(last_err());
        }
        set_sockopt_flag(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR);
        if reuseport {
            set_sockopt_flag(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT);
        }
        let addr = libc::sockaddr_in {
            sin_family: libc::AF_INET as libc::sa_family_t,
            sin_port: port.to_be(),
            sin_addr: libc::in_addr {
                s_addr: libc::INADDR_ANY.to_be(),
            },
            sin_zero: [0; 8],
        };
        if libc::bind(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ) < 0
        {
            let e = last_err();
            close_fd(fd);
            return Err(e);
        }
        if libc::listen(fd, backlog) < 0 {
            let e = last_err();
            close_fd(fd);
            return Err(e);
        }
        Ok(fd)
    }
}

/// Convenience: a plain (non-REUSEPORT) TCP listener.
pub fn tcp_listener(port: u16, backlog: i32) -> io::Result<RawFd> {
    tcp_listener_opts(port, backlog, false)
}

#[inline]
pub fn set_tcp_defer_accept(fd: RawFd, seconds: libc::c_int) {
    unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_DEFER_ACCEPT,
            &seconds as *const _ as *const c_void,
            mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

/// Creates a listening Unix-domain socket at `path`.
pub fn uds_listener(path: &Path, backlog: i32) -> io::Result<RawFd> {
    uds_listener_kind(path, backlog, libc::SOCK_STREAM)
}

/// Creates a listening sequenced-packet Unix-domain socket at `path`.
pub fn uds_seqpacket_listener(path: &Path, backlog: i32) -> io::Result<RawFd> {
    uds_listener_kind(path, backlog, libc::SOCK_SEQPACKET)
}

fn uds_listener_kind(path: &Path, backlog: i32, kind: libc::c_int) -> io::Result<RawFd> {
    let _ = std::fs::remove_file(path);
    unsafe {
        let fd = libc::socket(libc::AF_UNIX, kind, 0);
        if fd < 0 {
            return Err(last_err());
        }
        let mut addr: libc::sockaddr_un = mem::zeroed();
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let bytes = path.to_str().unwrap().as_bytes();
        if bytes.len() >= addr.sun_path.len() {
            close_fd(fd);
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "uds path too long",
            ));
        }
        for (i, b) in bytes.iter().enumerate() {
            addr.sun_path[i] = *b as libc::c_char;
        }
        let len = (mem::size_of::<libc::sa_family_t>() + bytes.len() + 1) as libc::socklen_t;
        if libc::bind(fd, &addr as *const _ as *const libc::sockaddr, len) < 0 {
            let e = last_err();
            close_fd(fd);
            return Err(e);
        }
        if libc::listen(fd, backlog) < 0 {
            let e = last_err();
            close_fd(fd);
            return Err(e);
        }
        Ok(fd)
    }
}

/// Connects (blocking) to a Unix-domain socket at `path`.
pub fn uds_connect(path: &Path) -> io::Result<RawFd> {
    uds_connect_kind(path, libc::SOCK_STREAM)
}

/// Connects (blocking) to a sequenced-packet Unix-domain socket at `path`.
pub fn uds_seqpacket_connect(path: &Path) -> io::Result<RawFd> {
    uds_connect_kind(path, libc::SOCK_SEQPACKET)
}

fn uds_connect_kind(path: &Path, kind: libc::c_int) -> io::Result<RawFd> {
    unsafe {
        let fd = libc::socket(libc::AF_UNIX, kind, 0);
        if fd < 0 {
            return Err(last_err());
        }
        let mut addr: libc::sockaddr_un = mem::zeroed();
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let bytes = path.to_str().unwrap().as_bytes();
        if bytes.len() >= addr.sun_path.len() {
            close_fd(fd);
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "uds path too long",
            ));
        }
        for (i, b) in bytes.iter().enumerate() {
            addr.sun_path[i] = *b as libc::c_char;
        }
        let len = (mem::size_of::<libc::sa_family_t>() + bytes.len() + 1) as libc::socklen_t;
        if libc::connect(fd, &addr as *const _ as *const libc::sockaddr, len) < 0 {
            let e = last_err();
            close_fd(fd);
            return Err(e);
        }
        Ok(fd)
    }
}

/// Accepts one connection (blocking) from a listener.
pub fn accept(listen_fd: RawFd) -> io::Result<RawFd> {
    unsafe {
        let fd = libc::accept(listen_fd, std::ptr::null_mut(), std::ptr::null_mut());
        if fd < 0 {
            return Err(last_err());
        }
        Ok(fd)
    }
}

/// Accepts one connection (non-blocking listener); WouldBlock when drained.
pub fn accept_nb(listen_fd: RawFd) -> Io {
    unsafe {
        let fd = libc::accept4(
            listen_fd,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            libc::SOCK_NONBLOCK,
        );
        if fd >= 0 {
            return Io::Ok(fd as usize);
        }
        let e = last_err();
        match e.raw_os_error() {
            Some(libc::EAGAIN) => Io::WouldBlock,
            _ => Io::Err(e),
        }
    }
}

// Aligned scratch for a single-fd SCM_RIGHTS control message.
#[repr(C)]
union CmsgBuf {
    _align: libc::cmsghdr,
    buf: [u8; 32],
}

/// Sends `fd` to the peer over a connected Unix socket via SCM_RIGHTS.
pub fn send_fd(channel: RawFd, fd: RawFd) -> io::Result<()> {
    send_fd_flags(channel, fd, libc::MSG_NOSIGNAL)
}

/// Sends `fd` without blocking the control channel.
pub fn send_fd_nonblocking(channel: RawFd, fd: RawFd) -> io::Result<()> {
    send_fd_flags(channel, fd, libc::MSG_NOSIGNAL | libc::MSG_DONTWAIT)
}

fn send_fd_flags(channel: RawFd, fd: RawFd, flags: libc::c_int) -> io::Result<()> {
    unsafe {
        let mut byte: u8 = 0;
        let mut iov = libc::iovec {
            iov_base: &mut byte as *mut _ as *mut c_void,
            iov_len: 1,
        };
        let mut cmsg = CmsgBuf { buf: [0; 32] };
        let mut msg: libc::msghdr = mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg.buf.as_mut_ptr() as *mut c_void;
        msg.msg_controllen = libc::CMSG_SPACE(mem::size_of::<RawFd>() as u32) as _;

        let chdr = libc::CMSG_FIRSTHDR(&msg);
        (*chdr).cmsg_level = libc::SOL_SOCKET;
        (*chdr).cmsg_type = libc::SCM_RIGHTS;
        (*chdr).cmsg_len = libc::CMSG_LEN(mem::size_of::<RawFd>() as u32) as _;
        std::ptr::copy_nonoverlapping(
            &fd as *const RawFd as *const u8,
            libc::CMSG_DATA(chdr),
            mem::size_of::<RawFd>(),
        );

        loop {
            let n = libc::sendmsg(channel, &msg, flags);
            if n >= 0 {
                return Ok(());
            }
            let e = last_err();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(e);
        }
    }
}

/// Outcome of a non-blocking `recv_fd`.
pub enum RecvFd {
    /// A passed file descriptor.
    Fd(RawFd),
    /// No more messages queued right now.
    WouldBlock,
    /// The control channel was closed by the peer.
    Closed,
}

/// Receives one fd from a connected Unix socket via SCM_RIGHTS (non-blocking).
/// Callers should loop until `WouldBlock` to drain the channel promptly — the
/// LB blocks on `sendmsg` once the in-flight fd buffer fills (Linux caps it).
pub fn recv_fd(channel: RawFd) -> RecvFd {
    unsafe {
        let mut byte: u8 = 0;
        let mut iov = libc::iovec {
            iov_base: &mut byte as *mut _ as *mut c_void,
            iov_len: 1,
        };
        let mut cmsg = CmsgBuf { buf: [0; 32] };
        let mut msg: libc::msghdr = mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg.buf.as_mut_ptr() as *mut c_void;
        msg.msg_controllen = 32;

        loop {
            let n = libc::recvmsg(channel, &mut msg, libc::MSG_DONTWAIT);
            if n == 0 {
                return RecvFd::Closed;
            }
            if n < 0 {
                let e = last_err();
                match e.raw_os_error() {
                    Some(libc::EINTR) => continue,
                    Some(libc::EAGAIN) => return RecvFd::WouldBlock,
                    _ => return RecvFd::Closed,
                }
            }
            let chdr = libc::CMSG_FIRSTHDR(&msg);
            if chdr.is_null() {
                // Data byte without a control message — ignore, try next.
                return RecvFd::WouldBlock;
            }
            if (*chdr).cmsg_level == libc::SOL_SOCKET && (*chdr).cmsg_type == libc::SCM_RIGHTS {
                let mut fd: RawFd = -1;
                std::ptr::copy_nonoverlapping(
                    libc::CMSG_DATA(chdr),
                    &mut fd as *mut RawFd as *mut u8,
                    mem::size_of::<RawFd>(),
                );
                return RecvFd::Fd(fd);
            }
            return RecvFd::WouldBlock;
        }
    }
}

/// Non-blocking read into `buf`.
#[inline]
pub fn read(fd: RawFd, buf: &mut [u8]) -> Io {
    unsafe {
        let n = libc::read(fd, buf.as_mut_ptr() as *mut c_void, buf.len());
        if n > 0 {
            return Io::Ok(n as usize);
        }
        if n == 0 {
            return Io::Eof;
        }
        let e = last_err();
        match e.raw_os_error() {
            Some(libc::EAGAIN) => Io::WouldBlock,
            Some(libc::EINTR) => read(fd, buf),
            _ => Io::Err(e),
        }
    }
}

/// Best-effort full write of `buf` (loops on partial/EINTR). Small responses
/// over a fresh socket essentially never block here.
pub fn write_all(fd: RawFd, buf: &[u8]) -> io::Result<()> {
    let mut off = 0;
    while off < buf.len() {
        let n = unsafe { libc::write(fd, buf[off..].as_ptr() as *const c_void, buf.len() - off) };
        if n > 0 {
            off += n as usize;
            continue;
        }
        let e = last_err();
        match e.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EAGAIN) => {
                // Rare for tiny responses; spin briefly.
                continue;
            }
            _ => return Err(e),
        }
    }
    Ok(())
}

/// Locks `len` bytes at `ptr` into RAM (no page faults during queries).
pub fn mlock(ptr: *const u8, len: usize) {
    unsafe {
        libc::mlock(ptr as *const c_void, len);
    }
}

/// Hints the kernel that the mmap'd index should be backed eagerly.
pub fn madvise_willneed(ptr: *const u8, len: usize) {
    unsafe {
        libc::madvise(ptr as *mut c_void, len, libc::MADV_WILLNEED);
    }
}

/// Hints the kernel that transparent huge pages may help the mmap'd index.
pub fn madvise_hugepage(ptr: *const u8, len: usize) {
    unsafe {
        libc::madvise(ptr as *mut c_void, len, libc::MADV_HUGEPAGE);
    }
}

// ---------------------------------------------------------------------------
// epoll
// ---------------------------------------------------------------------------

pub struct Epoll {
    pub fd: RawFd,
}

#[repr(C)]
struct EpollBusyPollParams {
    busy_poll_usecs: u32,
    busy_poll_budget: u16,
    prefer_busy_poll: u8,
    _pad: u8,
}

const fn ioctl_iow(ty: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((1u32 << 30) | (size << 16) | (ty << 8) | nr) as libc::c_ulong
}

const EPIOCSPARAMS: libc::c_ulong = ioctl_iow(
    0x8A,
    0x01,
    std::mem::size_of::<EpollBusyPollParams>() as u32,
);

impl Epoll {
    pub fn new() -> io::Result<Self> {
        let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if fd < 0 {
            return Err(last_err());
        }
        Ok(Epoll { fd })
    }

    pub fn set_busy_poll(&self, usecs: u32, budget: u16, prefer: bool) {
        if usecs == 0 && !prefer {
            return;
        }
        let params = EpollBusyPollParams {
            busy_poll_usecs: usecs,
            busy_poll_budget: budget,
            prefer_busy_poll: prefer as u8,
            _pad: 0,
        };
        unsafe {
            libc::ioctl(self.fd, EPIOCSPARAMS, &params as *const EpollBusyPollParams);
        }
    }

    pub fn add(&self, fd: RawFd, data: u64) -> io::Result<()> {
        let mut ev = libc::epoll_event {
            events: libc::EPOLLIN as u32,
            u64: data,
        };
        let r = unsafe { libc::epoll_ctl(self.fd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
        if r < 0 {
            return Err(last_err());
        }
        Ok(())
    }

    pub fn del(&self, fd: RawFd) {
        unsafe {
            libc::epoll_ctl(self.fd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut());
        }
    }

    /// Waits for events. `timeout_ms < 0` blocks indefinitely.
    pub fn wait(&self, events: &mut [libc::epoll_event], timeout_ms: i32) -> io::Result<usize> {
        loop {
            let n = unsafe {
                libc::epoll_wait(
                    self.fd,
                    events.as_mut_ptr(),
                    events.len() as i32,
                    timeout_ms,
                )
            };
            if n >= 0 {
                return Ok(n as usize);
            }
            let e = last_err();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(e);
        }
    }

    /// Waits with microsecond precision on kernels that support epoll_pwait2,
    /// falling back to a millisecond epoll_wait timeout otherwise.
    pub fn wait_micros(
        &self,
        events: &mut [libc::epoll_event],
        timeout_us: u64,
    ) -> io::Result<usize> {
        let ts = libc::timespec {
            tv_sec: (timeout_us / 1_000_000) as libc::time_t,
            tv_nsec: ((timeout_us % 1_000_000) * 1000) as libc::c_long,
        };
        loop {
            let n = unsafe {
                libc::epoll_pwait2(
                    self.fd,
                    events.as_mut_ptr(),
                    events.len() as i32,
                    &ts,
                    std::ptr::null(),
                )
            };
            if n >= 0 {
                return Ok(n as usize);
            }
            let e = last_err();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            if e.raw_os_error() == Some(libc::ENOSYS) {
                let ms = ((timeout_us + 999) / 1000).min(i32::MAX as u64) as i32;
                return self.wait(events, ms);
            }
            return Err(e);
        }
    }
}
