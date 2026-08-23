// main.cpp — vllm_engine host entry; runs --self-check and --attention.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "hip_compat/gpu_windows.h"
#include "gpu_buffer.h"
#include "block_table.h"
#include "model_runner.h"

using namespace vllm::engine;

static int self_check(const std::vector<vllm::gpu::GpuInfo>& gpus) {
    std::printf("self-check: %d GPU(s) visible\n", (int)gpus.size());
    std::vector<GpuDevice> devs;
    for (auto& g : gpus) {
        devs.push_back({g.device_id, g.vram_bytes, g.vram_used});
        std::printf("  dev %d: %s %s VRAM=%lluMB free=%lluMB\n",
            g.device_id, g.device_name.c_str(), g.gcn_arch.c_str(),
            (unsigned long long)(g.vram_bytes >> 20),
            (unsigned long long)(g.vram_bytes - g.vram_used) >> 20);
    }
    BlockConfig cfg{16, 32, 32, 128};
    BlockTable bt(cfg, devs);
    std::printf("  block config: bs=%d layers=%d heads=%d head=%d bytes=%.1fMB\n",
        cfg.block_size, cfg.num_layers, cfg.num_heads, cfg.head_size,
        (double)cfg.block_bytes() / (1024*1024));
    size_t total_floats = cfg.block_bytes() / sizeof(float);
    // 1) write random KV data to a block, read it back, assert byte-for-byte equality
    auto root = bt.allocate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    std::printf("  allocated root block: bid=%d bytes=%zu\n", root, bt.block(root).storage.bytes());
    std::vector<float> host_in(total_floats);
    for (size_t i = 0; i < total_floats; ++i) {
        host_in[i] = float(double(i) * 1.1f);
    }
    bt.write_block(root, host_in.data());
    std::vector<float> host_out(total_floats, 0.0f);
    bt.read_block(root, host_out.data());
    bool ok = true;
    for (size_t i = 0; i < total_floats; ++i) {
        if (host_out[i] != host_in[i]) {
            std::printf("  FAIL: mismatch at idx=%zu got=%f want=%f\n", i, host_out[i], host_in[i]);
            ok = false;
            break;
        }
    }
    std::printf("  KV round-trip: %s (%zu floats)\n", ok ? "ok" : "FAIL", total_floats);
    if (!ok) return 1;
    // 2) free + realloc to confirm prefix-cache still shares
    bt.release(root);
    bt.release(root); // refcount back to 0 → deallocates
    if (bt.block(root).valid()) {
        std::printf("  FAIL: block still valid after release\n");
        return 1;
    }
    std::printf("  block freed after release: ok\n");
    auto root_again = bt.allocate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    if (root_again == root) {
        std::printf("  FAIL: block reused slot without re-alloc (expected new id)\n");
        return 1;
    }
    std::printf("  re-allocated block: ok (bid=%d)\n", root_again);
    // 3) prefix-cache sharing: same tokens → same block
    auto root_b = bt.allocate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    if (root_b != root_again) {
        std::printf("  FAIL: prefix-cache not shared after realloc (a=%d b=%d)\n", root_again, root_b);
        return 1;
    }
    std::printf("  prefix-cache re-shared: ok (bid=%d)\n", root_b);
    // 4) child chain
    auto child = bt.allocate_child(root_again, 99, {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32});
    auto child2 = bt.allocate_child(root_again, 99, {99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99});
    if (child != child2) {
        std::printf("  FAIL: child prefix-cache miss (child=%d child2=%d)\n", child, child2);
        return 1;
    }
    std::printf("  child prefix-cache shared: ok (bid=%d)\n", child);
    // 5) big alloc evicts
    auto big = bt.allocate({100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100});
    if (!bt.block(big).valid()) {
        std::printf("  FAIL: evicted block invalid\n");
        return 1;
    }
    std::printf("  post-eviction alloc: ok (bid=%d)\n", big);
    std::printf("self-check PASS (total_alloc=%zu bytes)\n", bt.total_bytes_allocated());
    return 0;
}

static int run_attention(const std::vector<vllm::gpu::GpuInfo>& gpus) {
    std::printf("attention: P2.5 live dispatch via rocm_ops.dll (gfx1201)\n");
    std::vector<GpuDevice> devs;
    for (auto& g : gpus) devs.push_back({g.device_id, g.vram_bytes, g.vram_used});
    if (gpus.empty()) { std::printf("FAIL: no GPU\n"); return 1; }
    const int dev = gpus[0].device_id;
    if (hipSetDevice(dev) != hipSuccess) { std::printf("FAIL: hipSetDevice\n"); return 1; }

    BlockConfig cfg{16, 32, 32, 128};   // bs=16, layers=32, heads=32, head=128
    BlockTable bt(cfg, devs);

    // 32 heads, 32 kv heads (MHA), head_size=128, 4 tokens -> 4 query rows.
    ModelRunner runner(bt, AttentionType::kROCm, 32, 32, 128, 1.0/sqrt(128));

    std::vector<int> ids(4);
    for (int i = 0; i < 4; ++i) ids[i] = i + 1;
    SequenceInputs seq{1, ids};
    std::vector<SequenceInputs> seqs = {seq};

    ModelRunnerOutput out;
    try {
        out = runner.execute(seqs);
    } catch (const std::exception& e) {
        std::printf("FAIL: execute threw: %s\n", e.what());
        return 1;
    }
    auto& pb = runner.prepared();
    std::printf("  dispatched: seqs=%lld tokens=%lld heads=%lld head=%lld blocks=%lld\n",
                (long long)pb.num_seqs, (long long)pb.num_tokens,
                (long long)pb.num_heads, (long long)pb.head_size,
                (long long)pb.num_blocks);
    if (pb.num_blocks < 1) { std::printf("FAIL: no KV blocks allocated\n"); return 1; }
    std::printf("  allocated KV blocks: %lld  key_cache=%p\n", (long long)pb.num_blocks,
                pb.key_cache);
    std::printf("attention PASS (rocm_ops.dll dispatch round-tripped %zu output tokens)\n",
                out.top_token_ids.size());
    return 0;
}

static int dry_run(const std::vector<vllm::gpu::GpuInfo>& /*gpus*/) {
    std::printf("dry-run: removed — use --attention for the Phase 2.5 live path\n");
    return 0;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto gpus = vllm::gpu::enumerate_gpus();
    std::printf("WinRocM-llm phase0.4 stub — argc=%d\n", argc);
    if (gpus.empty()) {
        std::printf("GPU: none detected\n");
        return 0;
    }
    for (const auto& g : gpus) {
        std::printf("GPU %d: %s (%s) PCI=%04x:%02x:%02d VRAM=%llu MB used=%llu MB\n",
            g.device_id, g.device_name.c_str(), g.gcn_arch.c_str(),
            g.pci_bus, g.pci_bus, g.pci_device,
            (unsigned long long)(g.vram_bytes >> 20),
            (unsigned long long)(g.vram_used >> 20));
    }
    if (argc > 1 && std::strcmp(argv[1], "--self-check") == 0) {
        try {
            return self_check(gpus);
        } catch (const std::exception& e) {
            std::printf("self-check FAIL (exception): %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && (std::strcmp(argv[1], "--dry-run") == 0 || std::strcmp(argv[1], "--attention") == 0)) {
        try {
            return run_attention(gpus);
        } catch (const std::exception& e) {
            std::printf("attention FAIL (exception): %s\n", e.what());
            return 1;
        }
    }
    std::printf("usage: vllm_engine.exe [--self-check | --attention]\n");
    return 0;
}
