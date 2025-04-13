#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    upcxx::global_ptr<kmer_pair>* data_ptrs;
    upcxx::global_ptr<int>* used_ptrs;
    upcxx::atomic_domain<int> atomic_flags;
    int slots_per_rank;

    struct SlotLoc {
        int rank;
        int offset;
    };

    size_t my_size;

    size_t size() const noexcept;

    HashMap(size_t size);

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions
    SlotLoc locate_slot(uint64_t slot) const;

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    upcxx::future<int> request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size) {
    int ranks = upcxx::rank_n();
    my_size = size;
    slots_per_rank = size / ranks + 1;
    
    // allocate distributed arrays
    upcxx::dist_object<upcxx::global_ptr<kmer_pair>> data(upcxx::new_array<kmer_pair>(slots_per_rank));
    upcxx::dist_object<upcxx::global_ptr<int>> used(upcxx::new_array<int>(slots_per_rank));
    
    // allocate pointer arrays for each rank
    data_ptrs = new upcxx::global_ptr<kmer_pair>[ranks];
    used_ptrs = new upcxx::global_ptr<int>[ranks];

    upcxx::barrier();

    // gather pointers from all ranks
    for (int r = 0; r < ranks; ++r) {
        data_ptrs[r] = data.fetch(r).wait();
        used_ptrs[r] = used.fetch(r).wait();
    }

    upcxx::barrier();
    
    // set up atomic domain for slot management
    atomic_flags = upcxx::atomic_domain<int>({
        upcxx::atomic_op::load,
        upcxx::atomic_op::fetch_add
    });
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    for (uint64_t probe = 0; probe < size(); ++probe) {
        uint64_t slot = (hash + probe) % size();
        if (request_slot(slot).wait() == 0) {
            write_slot(slot, kmer);
            return true;
        }
    }
    return false;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        val_kmer = read_slot(slot);
        if (val_kmer.kmer == key_kmer) {
            success = true;
        }
    } while (!success && probe < size());
    return success;
}

HashMap::SlotLoc HashMap::locate_slot(uint64_t slot) const {
    return {static_cast<int>(slot / slots_per_rank), static_cast<int>(slot % slots_per_rank)};
}

bool HashMap::slot_used(uint64_t slot) {
    auto loc = locate_slot(slot);
    upcxx::global_ptr<int> ptr = used_ptrs[loc.rank] + loc.offset;
    return atomic_flags.load(ptr, std::memory_order_relaxed).wait() != 0;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) {
    auto loc = locate_slot(slot);
    (void) upcxx::rput(kmer, data_ptrs[loc.rank] + loc.offset);
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    auto loc = locate_slot(slot);
    return upcxx::rget(data_ptrs[loc.rank] + loc.offset).wait();
}

upcxx::future<int> HashMap::request_slot(uint64_t slot) {
    auto loc = locate_slot(slot);
    upcxx::global_ptr<int> ptr = used_ptrs[loc.rank] + loc.offset;
    return atomic_flags.fetch_add(ptr, 1, std::memory_order_relaxed);
}

size_t HashMap::size() const noexcept { return my_size; }
