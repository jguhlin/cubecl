use crate::{
    memory_management::{
        BytesFormat, ManagedMemoryHandle, MemoryLocation, MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool, Slice},
    },
    server::IoError,
    storage::StorageId,
};
use alloc::vec::Vec;
use core::fmt::Display;

pub struct SlicedPool {
    /// Pages stored as Option to support tombstone-based deallocation.
    /// This keeps page indices stable so outstanding bindings remain valid.
    pages: Vec<Option<(MemoryPage, StorageId)>>,
    page_size: u64,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
}

impl SlicedPool {
    pub fn new(page_size: u64, max_slice_size: u64, alignment: u64, pool_pos: u8) -> Self {
        Self {
            pages: Vec::new(),
            page_size,
            alignment,
            max_alloc_size: max_slice_size,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
        }
    }
}

impl MemoryPool for SlicedPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
            ||
            // If the size is close to the page size so it doesn't create much fragmentation with
            // unused space.
            match self.page_size.checked_sub(size) {
                Some(diff) => diff * 5 < self.page_size, // 20 % unused space is the max allowed.
                None => false,
            }
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let page_index = binding.descriptor().page();
        let (page, _) = self
            .pages
            .get(page_index)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| IoError::NotFound {
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
                reason: alloc::format!(
                    "SlicedPool: page {} doesn't exist (tombstone or out of range)",
                    page_index
                )
                .into(),
            })?;
        page.find(binding)
    }

    fn try_reserve(&mut self, size: u64) -> Option<super::ManagedMemoryHandle> {
        for slot in self.pages.iter_mut() {
            if let Some((page, _)) = slot.as_mut() {
                page.coalesce();
                if let Some(handle) = page.try_reserve(size) {
                    return Some(handle);
                }
            }
        }

        None
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<super::ManagedMemoryHandle, crate::server::IoError> {
        let storage = storage.alloc(self.page_size)?;

        let storage_id = storage.id;

        // Find a tombstone slot or append.
        let page_index = if let Some(idx) = self.pages.iter().position(|s| s.is_none()) {
            idx
        } else {
            self.pages.push(None);
            self.pages.len() - 1
        };

        let mut location_base = self.location_base;
        location_base.page = page_index as u16;

        let mut page = MemoryPage::new(storage, self.alignment, location_base);
        let returned = page.try_reserve(size);
        self.pages[page_index] = Some((page, storage_id));

        Ok(returned.expect("effective_size to be smaller than page_size"))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for slot in self.pages.iter() {
            if let Some((page, _)) = slot {
                let current = page.memory_usage();
                usage = usage.combine(current);
            }
        }

        usage
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if !explicit {
            return;
        }

        // Deallocate in-place using tombstones to keep page indices stable.
        for slot in self.pages.iter_mut() {
            if let Some((page, id)) = slot.as_mut() {
                page.coalesce();
                let summary = page.summary(false);

                if summary.amount_free == summary.amount_total {
                    storage.dealloc(*id);
                    *slot = None; // tombstone
                }
            }
        }

        // Trim trailing tombstones to avoid unbounded growth.
        while self.pages.last().map_or(false, |s| s.is_none()) {
            self.pages.pop();
        }
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let page_index = reserved.descriptor().page();
        let (page, _) = self
            .pages
            .get_mut(page_index)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| IoError::NotFound {
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
                reason: alloc::format!(
                    "SlicedPool: page {} doesn't exist for bind (tombstone or out of range)",
                    page_index
                )
                .into(),
            })?;

        page.bind(reserved, assigned, cursor)?;

        Ok(())
    }
}

impl Display for SlicedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if !self.pages.iter().any(|s| s.is_some()) {
            return Ok(());
        }

        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}\n",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.max_alloc_size)
        ))?;

        for (i, slot) in self.pages.iter().enumerate() {
            if let Some((page, id)) = slot {
                let summary = page.summary(false);
                f.write_fmt(format_args!(
                    "   - Page[{i}] {id} num_slices={} =>",
                    summary.num_total
                ))?;

                let size_free = BytesFormat::new(summary.amount_free);
                let size_full = BytesFormat::new(summary.amount_full);
                let size_total = BytesFormat::new(summary.amount_total);

                f.write_fmt(format_args!(
                    " {size_free} free - {size_full} full - {size_total} total\n"
                ))?;
            }
        }

        f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;

        Ok(())
    }
}
