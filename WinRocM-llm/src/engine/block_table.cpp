#include "block_table.h"
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace vllm::engine {

static uint64_t hash_tokens(const std::vector<int>& ids) {
    std::hash<int> hasher;
    std::size_t h = 0;
    for (int x : ids) h = hasher(h ^ (x + 0x9e3779b9 + (h << 6) + (h >> 2)));
    return h;
}

BlockTable::BlockTable(BlockConfig cfg, std::vector<GpuDevice> devices)
    : cfg_(cfg), devices_(std::move(devices)) {
    if (devices_.empty()) {
        throw std::runtime_error("BlockTable: no GPU devices");
    }
    blocks_.reserve(256);
    blocks_.emplace_back(); // index 0 = sentinel invalid
}

int BlockTable::device_for_block(size_t bytes, int hint) {
    if (hint >= 0 && hint < static_cast<int>(devices_.size()) &&
        devices_[hint].free_bytes() >= bytes) {
        return hint;
    }
    // best-fit: smallest remaining free space that still fits (minimizes fragmentation).
    int best = -1;
    size_t best_free = SIZE_MAX;
    for (size_t i = 0; i < devices_.size(); ++i) {
        size_t f = devices_[i].free_bytes();
        if (f >= bytes && f < best_free) {
            best_free = f;
            best = static_cast<int>(i);
        }
    }
    if (best < 0) throw std::runtime_error("BlockTable: no device has capacity");
    return best;
}

BlockKey BlockTable::make_key(int parent_id, uint64_t block_hash) const {
    return static_cast<BlockKey>((uint64_t(parent_id) << 32) | (block_hash & 0xFFFFFFFFu));
}

std::optional<int> BlockTable::find_shared(int parent_id, uint64_t block_hash) const {
    uint64_t key = (uint64_t(parent_id) << 32) | (block_hash & 0xFFFFFFFFu);
    auto it = prefix_index_.find(key);
    if (it == prefix_index_.end() || it->second.empty()) return std::nullopt;
    for (int bid : it->second) {
        if (blocks_[bid].valid() && blocks_[bid].ref_count > 0) return bid;
    }
    return std::nullopt;
}

int BlockTable::allocate(const std::vector<int>& token_ids, int device_hint) {
    uint64_t hash = hash_tokens(token_ids);
    if (auto shared = find_shared(-1, hash)) return *shared;
    size_t bytes = cfg_.block_bytes();
    int dev = device_for_block(bytes, device_hint);
    int bid = next_block_id_++;
    blocks_.emplace_back();
    blocks_[bid] = Block{make_key(-1, hash), hash, -1, dev, bid,
                         GPUBuffer(bytes, dev), lru_.end(), 1};
    lru_.push_back(bid);
    blocks_[bid].lru_node = std::prev(lru_.end());
    devices_[dev].vram_used += bytes;
    total_allocated_ += bytes;
    prefix_index_[static_cast<uint64_t>(blocks_[bid].key)].push_back(bid);
    return bid;
}

int BlockTable::allocate_child(int parent_id, uint64_t block_hash,
                               const std::vector<int>& token_ids) {
    if (auto shared = find_shared(parent_id, block_hash)) {
        blocks_[*shared].ref_count += 1;
        lru_.erase(blocks_[*shared].lru_node);
        lru_.push_back(*shared);
        blocks_[*shared].lru_node = std::prev(lru_.end());
        return *shared;
    }
    if (parent_id < 0 || parent_id >= next_block_id_ || !blocks_[parent_id].valid()) {
        throw std::runtime_error("BlockTable::allocate_child: invalid parent");
    }
    size_t bytes = cfg_.block_bytes();
    int dev = blocks_[parent_id].device_id;
    int bid = next_block_id_++;
    blocks_.emplace_back();
    blocks_[bid] = Block{make_key(parent_id, block_hash), hash_tokens(token_ids),
                         parent_id, dev, bid,
                         GPUBuffer(bytes, dev), lru_.end(), 1};
    lru_.push_back(bid);
    blocks_[bid].lru_node = std::prev(lru_.end());
    devices_[dev].vram_used += bytes;
    total_allocated_ += bytes;
    prefix_index_[static_cast<uint64_t>(blocks_[bid].key)].push_back(bid);
    return bid;
}

size_t BlockTable::evict(size_t need_bytes) {
    size_t freed = 0;
    auto it = lru_.begin();
    while (need_bytes - freed > 0 && it != lru_.end()) {
        int bid = *it;
        Block& b = blocks_[bid];
        if (b.ref_count > 0) { ++it; continue; } // can't evict referenced
        size_t bsz = b.storage.bytes();
        devices_[b.device_id].vram_used -= bsz;
        total_allocated_ -= bsz;
        b.storage.deallocate();
        // remove from prefix index
        auto& v = prefix_index_[static_cast<uint64_t>(b.key)];
        v.erase(std::remove(v.begin(), v.end(), bid), v.end());
        if (v.empty()) prefix_index_.erase(static_cast<uint64_t>(b.key));
        freed += bsz;
        it = lru_.erase(it);
        b.key = BlockKey::kInvalid;
    }
    return freed;
}

void BlockTable::release(int block_id) {
    if (block_id <= 0 || block_id >= next_block_id_) return;
    Block& b = blocks_[block_id];
    if (!b.valid()) return;
    b.ref_count = (b.ref_count > 0) ? b.ref_count - 1 : 0;
    lru_.erase(b.lru_node);
    lru_.push_back(block_id);
    b.lru_node = std::prev(lru_.end());
    if (b.ref_count == 0) {
        size_t bsz = b.storage.bytes();
        devices_[b.device_id].vram_used -= bsz;
        total_allocated_ -= bsz;
        b.storage.deallocate();
        auto& v = prefix_index_[static_cast<uint64_t>(b.key)];
        v.erase(std::remove(v.begin(), v.end(), block_id), v.end());
        if (v.empty()) prefix_index_.erase(static_cast<uint64_t>(b.key));
        b.key = BlockKey::kInvalid;
        lru_.pop_back(); // just moved to back, now remove (was the back)
    }
}

const Block& BlockTable::block(int block_id) const {
    if (block_id < 0 || block_id >= next_block_id_) {
        throw std::out_of_range("BlockTable::block: bad id");
    }
    return blocks_[block_id];
}

void BlockTable::write_block(int block_id, const float* kv_pair) {
    if (block_id <= 0 || block_id >= next_block_id_) {
        throw std::out_of_range("BlockTable::write_block: bad id");
    }
    const Block& b = blocks_[block_id];
    if (!b.valid() || !b.storage.valid()) {
        throw std::runtime_error("BlockTable::write_block: block not allocated");
    }
    size_t bytes = cfg_.block_bytes();
    hipError_t err = hipMemcpyHtoD(const_cast<void*>(b.storage.device_ptr()), kv_pair, bytes);
    if (err != hipSuccess) {
        throw std::runtime_error("BlockTable::write_block: hipMemcpyHtoD failed");
    }
}

void BlockTable::read_block(int block_id, float* out) const {
    if (block_id <= 0 || block_id >= next_block_id_) {
        throw std::out_of_range("BlockTable::read_block: bad id");
    }
    const Block& b = blocks_[block_id];
    if (!b.valid() || !b.storage.valid()) {
        throw std::runtime_error("BlockTable::read_block: block not allocated");
    }
    size_t bytes = cfg_.block_bytes();
    hipError_t err = hipMemcpyDtoH(out, const_cast<void*>(b.storage.device_ptr()), bytes);
    if (err != hipSuccess) {
        throw std::runtime_error("BlockTable::read_block: hipMemcpyDtoH failed");
    }
}

void* BlockTable::block_device_ptr(int block_id) const {
    if (block_id <= 0 || block_id >= next_block_id_) return nullptr;
    const Block& b = blocks_[block_id];
    return b.valid() && b.storage.valid() ? const_cast<void*>(b.storage.device_ptr()) : nullptr;
}

} // namespace vllm::engine
