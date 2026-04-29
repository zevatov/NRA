pub const MAGIC_BYTES: [u8; 4] = [0x4E, 0x52, 0x41, 0x00]; // "NRA\0"
pub const FORMAT_VERSION: u16 = 2;
pub const HEADER_SIZE: usize = 32;

/// NRA Binary Header Layout (32 bytes, Little-Endian):
///
/// ```text
/// Offset  Size  Field
/// 0x00    4     Magic bytes ("NRA\0")
/// 0x04    2     Format version (u16)
/// 0x06    2     Flags (bitfield, reserved)
/// 0x08    8     Manifest offset (u64) — always 32 for v1
/// 0x10    8     Manifest size in bytes (u64)
/// 0x18    8     Data section offset (u64)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NraHeader {
    pub version: u16,
    pub flags: u16,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub data_section_offset: u64,
}

impl NraHeader {
    pub fn new(manifest_size: u64, data_section_offset: u64) -> Self {
        Self {
            version: FORMAT_VERSION,
            flags: 0, // Reserved for future bitflags
            manifest_offset: HEADER_SIZE as u64,
            manifest_size,
            data_section_offset,
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC_BYTES);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.manifest_offset.to_le_bytes());
        buf[16..24].copy_from_slice(&self.manifest_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.data_section_offset.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Result<Self, &'static str> {
        if buf[0..4] != MAGIC_BYTES {
            return Err("Invalid NRA magic bytes");
        }
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version == 0 || version > FORMAT_VERSION {
            return Err("Unsupported NRA format version");
        }
        let flags = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let manifest_offset = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let manifest_size = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let data_section_offset = u64::from_le_bytes(buf[24..32].try_into().unwrap());

        if manifest_offset < HEADER_SIZE as u64 {
            return Err("Manifest offset is inside the header region");
        }

        Ok(Self {
            version,
            flags,
            manifest_offset,
            manifest_size,
            data_section_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let header = NraHeader::new(512, 544);
        let bytes = header.to_bytes();
        let parsed = NraHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, parsed);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"ZIP\0");
        assert!(NraHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_future_version() {
        let mut header = NraHeader::new(0, 32);
        header.version = 99;
        let bytes = header.to_bytes();
        assert!(NraHeader::from_bytes(&bytes).is_err());
    }
}
