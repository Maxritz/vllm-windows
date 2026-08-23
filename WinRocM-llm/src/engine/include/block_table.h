#pragma once
/**
@file engine/block_table
@brief LRU block allocator with prefix-caching awareness for KV-cache.
@details
Implements vLLM's block-manager semantics in pure C++ on top of GPUBuffer:
  - Fixed block size (tokens per block, e.g. 16).
  - LRU eviction: least-recently-used blocks are evicted first when no free
    block is available.
  - Prefix-caching: a block is keyed by its (parent_block_id, block_hash) so a
    shared prefix across sequences reuses GPU memory — the core memory win.
  - Multi-GPU: blocks are pinned to a device_id; allocation picks the device
    with the most free bytes (best-fit across GPUs via gpu_windows enum).
No stubs. On allocation failure (all GPUs exhausted) throws.

@see knowledge/engine/core.dox.md
@status done
*/
#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu_buffer.h"

namespace vllm::engine {

struct GpuDevice {
    int device_id;
    size_t vram_bytes;
    size_t vram_used;
    size_t free_bytes() const noexcept { return vram_bytes - vram_used; }
};

enum class BlockKey : uint64_t { kInvalid = 0 };

struct BlockConfig {
    int block_size;
    int num_layers;
    int num_heads;
    int head_size;
    size_t block_bytes() const noexcept {
        return size_t(num_layers) * 2 * block_size * num_heads * head_size * sizeof(float);
    }
};

struct Block {
    BlockKey key;
    uint64_t block_hash;
    int parent_id;
    int device_id;
    int block_id;
    GPUBuffer storage;
    std::list<int>::iterator lru_node;
    int ref_count;
    bool valid() const noexcept { return key != BlockKey::kInvalid; }
};

class BlockTable {
public:
    explicit BlockTable(BlockConfig cfg, std::vector<GpuDevice> devices);
    // Allocate a block for a fresh chain (no parent), hashing the token ids.
    int allocate(const std::vector<int>& token_ids, int device_hint = -1);
    // Allocate a child of `parent_id` whose content hashes to `block_hash`.
    int allocate_child(int parent_id, uint64_t block_hash, const std::vector<int>& token_ids);
    // LRU eviction to free at least `need_bytes` on some device. Returns bytes freed.
    size_t evict(size_t need_bytes);
    // Release a sequence's reference to a block; frees on zero refcount.
    void release(int block_id);
    // Prefix-cache hit: reuse existing block with matching (parent_id, block_hash).
    std::optional<int> find_shared(int parent_id, uint64_t block_hash) const;
    const Block& block(int block_id) const;
    std::vector<GpuDevice> devices() const { return devices_; }
    const BlockConfig& config() const noexcept { return cfg_; }
    size_t total_bytes_allocated() const noexcept { return total_allocated_; }

    void write_block(int block_id, const float* kv_pair);
    void read_block(int block_id, float* out) const;

    /// Device pointer of a live block's KV storage (for paged-attention indirection).
    void* block_device_ptr(int block_id) const;
    int num_blocks_total() const noexcept { return next_block_id_ - 1; }
    // Primary device backing the KV slab (device 0 currently).
    int gpu_device_id() const noexcept { return devices_.empty() ? 0 : devices_[0].device_id; }

private:
    int device_for_block(size_t bytes, int hint);
    BlockKey make_key(int parent_id, uint64_t block_hash) const;
    int next_block_id_ = 1;
    const BlockConfig cfg_;
    std::vector<GpuDevice> devices_;
    std::vector<Block> blocks_;
    std::list<int> lru_;
    std::unordered_map<uint64_t, std::vector<int>> prefix_index_;
    size_t total_allocated_ = 0;
};

} // namespace vllm::engine
