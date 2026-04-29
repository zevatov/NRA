use fuser::{
    FileAttr, FileType, Filesystem, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry, Request,
};
use libc::ENOENT;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::sync::Mutex;
use std::time::{Duration, UNIX_EPOCH};

use crate::beta_reader::BetaReader;

const TTL: Duration = Duration::from_secs(1);

struct Node {
    ino: u64,
    #[allow(dead_code)]
    name: String,
    is_dir: bool,
    size: u64,
    children: HashMap<String, u64>, // name -> ino
    file_id: Option<String>,        // BetaReader file ID
}

pub struct NraFuse {
    reader: Mutex<BetaReader>,
    nodes: HashMap<u64, Node>,
    /// Cache of fully decompressed file data by inode, to avoid re-decompressing
    /// on every read() call from the kernel (which sends 128KB chunks).
    file_cache: Mutex<HashMap<u64, Vec<u8>>>,
    file_cache_order: Mutex<Vec<u64>>,
    /// Current user/group IDs (captured at mount time)
    uid: u32,
    gid: u32,
}

const MAX_CACHED_FILES: usize = 32;

impl NraFuse {
    pub fn new(reader: BetaReader) -> Self {
        let mut nodes = HashMap::new();
        // Create root node (ino 1)
        nodes.insert(
            1,
            Node {
                ino: 1,
                name: "".to_string(),
                is_dir: true,
                size: 0,
                children: HashMap::new(),
                file_id: None,
            },
        );

        let file_ids = reader.file_ids();
        for (next_ino, id) in (2..).zip(file_ids.iter()) {
            let size = reader.file_size(id).unwrap_or(0);
            
            // For V1 of the FUSE adapter, we flatten everything into the root directory.
            // Slashes are replaced with underscores to prevent OS path confusion.
            let safe_name = id.replace("/", "_");
            
            let ino = next_ino;

            nodes.insert(
                ino,
                Node {
                    ino,
                    name: safe_name.clone(),
                    is_dir: false,
                    size,
                    children: HashMap::new(),
                    file_id: Some(id.to_string()),
                },
            );

            // Add to root
            if let Some(root) = nodes.get_mut(&1) {
                root.children.insert(safe_name, ino);
            }
        }

        Self {
            reader: Mutex::new(reader),
            nodes,
            file_cache: Mutex::new(HashMap::new()),
            file_cache_order: Mutex::new(Vec::new()),
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
        }
    }

    fn get_attr(&self, ino: u64) -> Option<FileAttr> {
        let node = self.nodes.get(&ino)?;
        Some(FileAttr {
            ino: node.ino,
            size: node.size,
            blocks: node.size.div_ceil(512),
            atime: UNIX_EPOCH,
            mtime: UNIX_EPOCH,
            ctime: UNIX_EPOCH,
            crtime: UNIX_EPOCH,
            kind: if node.is_dir { FileType::Directory } else { FileType::RegularFile },
            perm: if node.is_dir { 0o755 } else { 0o444 },
            nlink: if node.is_dir { 2 } else { 1 },
            uid: self.uid,
            gid: self.gid,
            rdev: 0,
            flags: 0,
            blksize: 4096,
        })
    }
}

impl Filesystem for NraFuse {
    fn lookup(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let name_str = name.to_string_lossy().to_string();
        
        if let Some(parent_node) = self.nodes.get(&parent)
            && let Some(&child_ino) = parent_node.children.get(&name_str)
                && let Some(attr) = self.get_attr(child_ino) {
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
        reply.error(ENOENT);
    }

    fn getattr(&mut self, _req: &Request, ino: u64, reply: ReplyAttr) {
        if let Some(attr) = self.get_attr(ino) {
            reply.attr(&TTL, &attr);
        } else {
            reply.error(ENOENT);
        }
    }

    fn readdir(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        if let Some(node) = self.nodes.get(&ino) {
            let mut entries = vec![
                (ino, FileType::Directory, "."),
                (1, FileType::Directory, ".."),
            ];

            for (name, &child_ino) in &node.children {
                let kind = if self.nodes.get(&child_ino).unwrap().is_dir {
                    FileType::Directory
                } else {
                    FileType::RegularFile
                };
                entries.push((child_ino, kind, name));
            }

            for (i, entry) in entries.into_iter().enumerate().skip(offset as usize) {
                if reply.add(entry.0, (i + 1) as i64, entry.1, entry.2) {
                    break;
                }
            }
            reply.ok();
        } else {
            reply.error(ENOENT);
        }
    }

    fn read(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        if let Some(node) = self.nodes.get(&ino)
            && let Some(file_id) = &node.file_id {
                // Check file cache first to avoid re-decompressing on every 128KB read call
                let mut cache = self.file_cache.lock().unwrap();
                if !cache.contains_key(&ino) {
                    let mut reader = self.reader.lock().unwrap();
                    if let Ok(data) = reader.read_file(file_id) {
                        // Evict oldest entry if cache is full
                        let mut order = self.file_cache_order.lock().unwrap();
                        if cache.len() >= MAX_CACHED_FILES
                            && let Some(oldest) = order.first().copied() {
                                cache.remove(&oldest);
                                order.remove(0);
                            }
                        order.push(ino);
                        cache.insert(ino, data);
                    } else {
                        reply.error(ENOENT);
                        return;
                    }
                }

                if let Some(data) = cache.get(&ino) {
                    let end = std::cmp::min(data.len(), (offset + size as i64) as usize);
                    let start = std::cmp::min(data.len(), offset as usize);
                    reply.data(&data[start..end]);
                    return;
                }
            }
        reply.error(ENOENT);
    }
}
