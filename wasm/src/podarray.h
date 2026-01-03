#include <cstdlib>
#include <type_traits>

template <typename T>
struct PodArray {
    static_assert(std::is_standard_layout<T>::value && std::is_trivially_copyable<T>::value, "T is not a POD");

    T* data;
    size_t size;

    PodArray(size_t n) : data((T*)malloc(sizeof(T) * n)), size(n) {
    }

    // Make the deletions explicit, do not rely on destructors.
    void freeData() {
        free(data);
    }

    T& operator[](size_t i) {
        if (i >= size) {
            __builtin_trap();
        }
        return data[i];
    }

    const T& operator[](size_t i) const {
        if (i >= size) {
            __builtin_trap();
        }
        return data[i];
    }

    // Make this property read-only. Embind does not play nicely with raw pointers, wrap everything in uintptr_t.
    uintptr_t dataPtr() const {
        return reinterpret_cast<uintptr_t>(data);
    }

    // Make this property read-only
    size_t getSize() const {
        return size;
    }
};
