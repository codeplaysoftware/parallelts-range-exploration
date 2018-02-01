#include "gtest/gtest.h"
#include "aligned_allocator.h"

struct AlignedAllocator : public testing::Test {};

TEST_F(AlignedAllocator, TestWithVector) {
  std::vector<float, aligned_allocator<float>> tmp(16);

  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(tmp.data()) % sizeof(float));
}

TEST_F(AlignedAllocator, VectorPageBoundary) {
  constexpr size_t alignment = 4096;
  std::vector<float, aligned_allocator<float, alignment>> tmp(alignment);

  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(tmp.data()) % alignment);
}

TEST_F(AlignedAllocator, TooMuch) {
  aligned_allocator<float> allocator;
  ASSERT_THROW(allocator.allocate(std::size_t(-1)), std::bad_alloc);
}
