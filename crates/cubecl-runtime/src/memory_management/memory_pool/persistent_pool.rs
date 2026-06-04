use super::{ManagedMemoryHandle, MemoryPool, Slice, calculate_padding};
use crate::memory_management::{BytesFormat, MemoryLocation};
use crate::storage::StorageUtilization;
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::backtrace::BackTrace;
use hashbrown::HashMap;

pub struct PersistentPool {
    /// Slices stored as Option to support tombstone-based deallocation.
    /// This keeps slice indices stable so outstanding bindings remain valid.
    slices: Vec<Option<Slice>>,
    sizes: HashMap<u64, Vec<usize>>,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
}

impl core::fmt::Display for PersistentPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (size, positions) in self.sizes.iter() {
            let mut num_free = 0;
            let mut num_full = 0;
            let total = positions.len();

            for pos in positions {
                if let Some(slice) = &self.slices[*pos] {
                    if slice.is_free() {
                        num_free += 1;
                    } else {
                        num_full += 1;
                    }
                }
            }

            f.write_fmt(format_args!(
                "  - Slices {} =>  {num_free} free - {num_full} full - {total} total\n",
                BytesFormat::new(*size)
            ))?;
        }

        if !self.sizes.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;
        }

        Ok(())
    }
}

impl PersistentPool {
    pub fn new(max_alloc_size: u64, alignment: u64, pool_pos: u8) -> Self {
        Self {
            slices: Vec::new(),
            sizes: HashMap::new(),
            max_alloc_size,
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
        }
    }

    pub fn has_size(&mut self, size: u64) -> bool {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;
        self.sizes.contains_key(&effective_size)
    }
}

impl MemoryPool for PersistentPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let slice_index = binding.descriptor().slice();

        self.slices
            .get(slice_index)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!(
                    "PersistentPool: slice {} doesn't exist (tombstone or out of range)",
                    slice_index
                )
                .into(),
            })
    }

    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        if let Some(positions) = self.sizes.get_mut(&effective_size) {
            for pos in positions {
                if let Some(slice) = &mut self.slices[*pos] {
                    if slice.is_free() {
                        slice.storage.utilization.size = size;
                        slice.storage.utilization.offset = 0;
                        return Some(slice.handle.clone());
                    }
                }
            }
        }

        None
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        let storage_handle = storage.alloc(effective_size)?;
        let mut slice = Slice::new(storage_handle, padding);
        slice.storage.utilization = StorageUtilization { offset: 0, size };
        let slice_id = slice.descriptor();

        // Find a tombstone slot or append.
        let slice_pos = if let Some(idx) = self.slices.iter().position(|s| s.is_none()) {
            idx
        } else {
            self.slices.push(None);
            self.slices.len() - 1
        };

        let mut location = self.location_base;
        location.slice = slice_pos as u32;
        slice_id.update_location(location);

        match self.sizes.get_mut(&effective_size) {
            Some(vals) => {
                vals.push(slice_pos);
            }
            None => {
                self.sizes.insert(effective_size, vec![slice_pos]);
            }
        }

        let handle = slice.handle.clone();
        self.slices[slice_pos] = Some(slice);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .slices
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|slice| slice.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|slice| slice.padding).sum(),
            bytes_reserved: self
                .slices
                .iter()
                .filter_map(|s| s.as_ref())
                .map(|slice| slice.effective_size())
                .sum(),
        }
    }

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {
            // Deallocate free slices in-place using tombstones to keep indices stable.
            for slot in self.slices.iter_mut() {
                let dealloc_id = slot
                    .as_ref()
                    .filter(|slice| slice.is_free())
                    .map(|slice| slice.storage.id);
                if let Some(id) = dealloc_id {
                    storage.dealloc(id);
                    *slot = None; // tombstone
                }
            }

            // Rebuild sizes map to exclude tombstoned positions.
            let mut sizes = HashMap::<u64, Vec<usize>>::new();
            for (i, slot) in self.slices.iter().enumerate() {
                if let Some(slice) = slot {
                    let effective_size = slice.effective_size();
                    match sizes.get_mut(&effective_size) {
                        Some(vals) => vals.push(i),
                        None => {
                            sizes.insert(effective_size, vec![i]);
                        }
                    }
                }
            }
            self.sizes = sizes;

            // Trim trailing tombstones.
            while self.slices.last().map_or(false, |s| s.is_none()) {
                self.slices.pop();
            }

            storage.flush();
        }
    }

    fn bind(
        &mut self,
        old: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let slice_index = old.descriptor().slice();
        let slice = self
            .slices
            .get_mut(slice_index)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!(
                    "PersistentPool: slice {} doesn't exist for bind",
                    slice_index
                )
                .into(),
            })?;

        new.descriptor()
            .update_location(old.descriptor().location());
        slice.cursor = cursor;
        slice.handle = new;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::memory_management::memory_pool::calculate_padding;
    use crate::storage::BytesStorage;

    use super::*;

    /// `alloc` and `try_reserve` must use the same `sizes` key (`size + padding`). See #1224.
    #[test_log::test]
    fn persistent_pool_try_reserve_reuses_slice_with_padding() {
        let mut storage = BytesStorage::default();
        let alignment = 4u64;
        let mut pool = PersistentPool::new(1024 * 1024, alignment, 0);

        let size = 1025u64;
        assert_ne!(
            calculate_padding(size, alignment),
            0,
            "test needs non-zero padding so alloc vs try_reserve keys differed pre-fix"
        );

        let handle = pool.alloc(&mut storage, size).expect("alloc");
        assert!(
            pool.try_reserve(size).is_none(),
            "slice must stay reserved while the handle is alive"
        );

        core::mem::drop(handle);

        assert!(
            pool.try_reserve(size).is_some(),
            "freed slice should be reusable"
        );
    }

    #[test_log::test]
    fn persistent_pool() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentPool::new(1024 * 1024, 4, 0);

        let result = pool.try_reserve(1024);
        assert!(result.is_none(), "No alloc yet");

        let alloc1 = pool.alloc(&mut storage, 1024);
        let result = pool.try_reserve(1024);
        assert!(result.is_none(), "No free slice yet, handle1 is alive");

        core::mem::drop(alloc1);
        let result = pool.try_reserve(1024);
        assert!(result.is_some(), "Handle1 is free to be reused.");
        core::mem::drop(result);

        let result = pool.try_reserve(1025);
        assert!(result.is_none(), "Not the same size.");

        let alloc2 = pool.alloc(&mut storage, 1024);
        let usage = pool.get_memory_usage();
        assert_eq!(usage.bytes_in_use, 1024);
        assert_eq!(usage.bytes_reserved, 2048);

        let result = pool.try_reserve(1024);
        let usage = pool.get_memory_usage();
        assert!(result.is_some(), "Handle1 is free to be reused.");
        assert_eq!(usage.bytes_in_use, 2048);
        assert_eq!(usage.bytes_reserved, 2048);

        core::mem::drop(alloc2);
        core::mem::drop(result);

        let usage = pool.get_memory_usage();
        assert_eq!(usage.bytes_in_use, 0);
        assert_eq!(usage.bytes_reserved, 2048);
    }
}
