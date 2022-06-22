#pragma once

#include <light/src/light.hpp>

template <typename T>
class WrappedArray {
    std::size_t nexti;
    const std::size_t maxCapacity;
    T* store;

public:
    WrappedArray(std::size_t maxSize, T* wrapped)
      : nexti(0),
        maxCapacity(maxSize),
        store(wrapped)
    {}

    ~WrappedArray() {}

    bool full() const { return nexti == maxCapacity; }
    constexpr std::size_t capacity() const { return maxCapacity; }
    bool empty() const { return nexti == 0; }
    std::size_t size() const { return nexti; }
    void push_back(const T& value) {
      store[nexti] = value; nexti += 1;
    }
    void pop_back() { nexti -= 1; }
    const T& back() const { return store[nexti - 1]; }
    T& back() { return store[nexti - 1]; }
    void clear() { nexti = 0; }
    const T& operator[] (std::size_t i) const { return store[i]; }
    T& operator[] (std::size_t i) { return store[i]; }
    void skip(std::size_t n = 1) { nexti += n; }
};

template <class T, class B>
WrappedArray<T> makeArrayWrapper(B& data) {
  // Access the per-ray contributions using a wrapper data structure. Order and
  // correpsondence to pixels in the framebuffers are implicit.
  const std::size_t maxContributions = data.size() / sizeof(light::Contribution);
  T* dataPtr = reinterpret_cast<T*>(&data[0]);
  return WrappedArray<T>(maxContributions, dataPtr);
}

template <class T>
bool resizeContributionArray(WrappedArray<T>& contributions) {
  // We need to find the end of the stack of contributions - skip through the stack of
  // contributions until the end marker also record whether the path makes any contribution
  // to the render as we go.
  bool pathContributes = false;

  while (true) {
    contributions.skip();
    const auto type = contributions.back().type;
    if (type == light::Contribution::Type::EMIT ||
        type == light::Contribution::Type::DEBUG ||
        type == light::Contribution::Type::ESCAPED) {
      pathContributes = true;
      // Path tracer always stops on first emitter so this is the end of the path.
      break;
    }
    if (type == light::Contribution::Type::END) {
      break;
    }
  }

  return pathContributes;
}
