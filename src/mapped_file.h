#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <stdexcept>

class MappedFile {
  int fd = -1;
  void *addr = nullptr;
  size_t fileSize = 0;
 public:
  MappedFile() = default;

  explicit MappedFile(const std::string &filename) {
    fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open file");
    }

    // Obtain the size of the file
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      close(fd);
      throw std::runtime_error("Failed to get file size");
    }
    fileSize = sb.st_size;

    // Map the file into memory
    addr = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to map the file");
    }
  }

  ~MappedFile() {
    if (addr) munmap(addr, fileSize);
    if (fd != -1) close(fd);
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  MappedFile(MappedFile &&other) noexcept: fd(other.fd), addr(other.addr), fileSize(other.fileSize) {
    other.addr = nullptr;
    other.fd = -1;
    other.fileSize = 0;
  }

  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this == &other) return *this;
    addr = other.addr;
    fileSize = other.fileSize;
    fd = other.fd;

    other.addr = nullptr;
    other.fileSize = 0;
    other.fd = -1;
    return *this;
  }

  [[nodiscard]] const char *data() const &{
    return reinterpret_cast<const char *>(addr);
  }

  [[nodiscard]] size_t size() const &{ return fileSize; }
};
