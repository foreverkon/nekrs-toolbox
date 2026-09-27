#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <adios2.h>
#include <hdf5.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr std::uint8_t VTK_HEXAHEDRON = 12;
constexpr std::size_t DEFAULT_READ_CHUNK = 5'000'000;
constexpr std::size_t DEFAULT_HDF_CHUNK_TUPLES = 1'000'000;
constexpr std::size_t DEFAULT_FULL_BLOCK_READ_LIMIT_MIB = 4096;
constexpr const char* STATE_GROUP = "_bp_post";
constexpr std::size_t MIN_COMPRESSED_ELEMENTS = 1024;

struct CompressionSettings
{
    bool enabled = true;
    unsigned int level = 1;
    unsigned int threads = 4;
    std::size_t chunkTuples = DEFAULT_HDF_CHUNK_TUPLES;
};

CompressionSettings gCompression;
std::size_t gFullBlockReadBytes = DEFAULT_FULL_BLOCK_READ_LIMIT_MIB << 20;

void H5Check(herr_t status, const std::string& operation)
{
    if (status < 0)
        throw std::runtime_error("HDF5 failure: " + operation);
}

hid_t H5Valid(hid_t id, const std::string& operation)
{
    if (id < 0)
        throw std::runtime_error("HDF5 failure: " + operation);
    return id;
}

std::size_t Product(const adios2::Dims& dims)
{
    return std::accumulate(dims.begin(), dims.end(), std::size_t{1}, std::multiplies<>());
}

hid_t CreateGroup(hid_t parent, const std::string& name)
{
    return H5Valid(H5Gcreate2(parent, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                   "create group " + name);
}

bool LinkExists(hid_t parent, const std::string& name)
{
    const htri_t result = H5Lexists(parent, name.c_str(), H5P_DEFAULT);
    if (result < 0)
        throw std::runtime_error("HDF5 failure: check link " + name);
    return result > 0;
}

hid_t OpenGroup(hid_t parent, const std::string& name)
{
    return H5Valid(H5Gopen2(parent, name.c_str(), H5P_DEFAULT), "open group " + name);
}

void WriteStringAttribute(hid_t object, const std::string& name, const std::string& value)
{
    hid_t type = H5Valid(H5Tcopy(H5T_C_S1), "copy string type");
    H5Check(H5Tset_size(type, value.size() + 1), "set string size");
    H5Check(H5Tset_strpad(type, H5T_STR_NULLTERM), "set string padding");
    hid_t space = H5Valid(H5Screate(H5S_SCALAR), "create scalar attribute space");
    hid_t attr = H5Valid(H5Acreate2(object, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT),
                         "create attribute " + name);
    H5Check(H5Awrite(attr, type, value.data()), "write attribute " + name);
    H5Aclose(attr); H5Sclose(space); H5Tclose(type);
}

std::string ReadStringAttribute(hid_t object, const std::string& name)
{
    hid_t attr = H5Valid(H5Aopen(object, name.c_str(), H5P_DEFAULT), "open " + name);
    hid_t type = H5Valid(H5Aget_type(attr), "get string type");
    std::vector<char> buffer(H5Tget_size(type) + 1, '\0');
    H5Check(H5Aread(attr, type, buffer.data()), "read " + name);
    H5Tclose(type); H5Aclose(attr);
    return buffer.data();
}

template <class T> hid_t NativeType();
template <> hid_t NativeType<float>() { return H5T_NATIVE_FLOAT; }
template <> hid_t NativeType<double>() { return H5T_NATIVE_DOUBLE; }
template <> hid_t NativeType<std::uint8_t>() { return H5T_NATIVE_UINT8; }
template <> hid_t NativeType<std::uint32_t>() { return H5T_NATIVE_UINT32; }
template <> hid_t NativeType<std::uint64_t>() { return H5T_NATIVE_UINT64; }
template <> hid_t NativeType<std::int64_t>() { return H5T_NATIVE_INT64; }
template <> hid_t NativeType<std::int32_t>() { return H5T_NATIVE_INT32; }

template <class T> hid_t FileType();
template <> hid_t FileType<float>() { return H5T_IEEE_F32LE; }
template <> hid_t FileType<double>() { return H5T_IEEE_F64LE; }
template <> hid_t FileType<std::uint8_t>() { return H5T_STD_U8LE; }
template <> hid_t FileType<std::uint32_t>() { return H5T_STD_U32LE; }
template <> hid_t FileType<std::uint64_t>() { return H5T_STD_U64LE; }
template <> hid_t FileType<std::int64_t>() { return H5T_STD_I64LE; }
template <> hid_t FileType<std::int32_t>() { return H5T_STD_I32LE; }

template <class T>
void WriteAttribute(hid_t object, const std::string& name, const std::vector<T>& values)
{
    hsize_t dim = values.size();
    hid_t space = H5Valid(H5Screate_simple(1, &dim, nullptr), "create attribute space");
    hid_t attr = H5Valid(H5Acreate2(object, name.c_str(), FileType<T>(), space,
                                    H5P_DEFAULT, H5P_DEFAULT), "create attribute " + name);
    H5Check(H5Awrite(attr, NativeType<T>(), values.data()), "write attribute " + name);
    H5Aclose(attr); H5Sclose(space);
}

template <class T>
void SetAttribute(hid_t object, const std::string& name, const std::vector<T>& values)
{
    if (!H5Aexists(object, name.c_str()))
    {
        WriteAttribute(object, name, values);
        return;
    }
    hid_t attr = H5Valid(H5Aopen(object, name.c_str(), H5P_DEFAULT),
                         "open attribute " + name);
    H5Check(H5Awrite(attr, NativeType<T>(), values.data()), "update attribute " + name);
    H5Aclose(attr);
}

template <class T>
hid_t CreateDataset(hid_t group, const std::string& name, const std::vector<hsize_t>& dims,
                    std::size_t chunkTuples = 0,
                    bool extensibleFirstDimension = false)
{
    if (!chunkTuples)
        chunkTuples = gCompression.chunkTuples;
    std::vector<hsize_t> maximum = dims;
    if (extensibleFirstDimension)
        maximum[0] = H5S_UNLIMITED;
    hid_t space = H5Valid(H5Screate_simple(static_cast<int>(dims.size()), dims.data(),
                                           extensibleFirstDimension ? maximum.data() : nullptr),
                          "create dataspace " + name);
    hid_t properties = H5Valid(H5Pcreate(H5P_DATASET_CREATE), "create dataset properties");
    std::vector<hsize_t> chunks = dims;
    chunks[0] = std::max<hsize_t>(1, std::min<hsize_t>(dims[0], chunkTuples));
    H5Check(H5Pset_chunk(properties, static_cast<int>(chunks.size()), chunks.data()),
            "set chunks " + name);
    const std::size_t elementCount = std::accumulate(
        dims.begin(), dims.end(), std::size_t{1},
        [](std::size_t left, hsize_t right) { return left * static_cast<std::size_t>(right); });
    if (gCompression.enabled && elementCount >= MIN_COMPRESSED_ELEMENTS)
    {
        H5Check(H5Pset_shuffle(properties), "enable shuffle filter for " + name);
        H5Check(H5Pset_deflate(properties, gCompression.level),
                "enable deflate filter for " + name);
    }
    hid_t dataset = H5Valid(H5Dcreate2(group, name.c_str(), FileType<T>(), space,
                                       H5P_DEFAULT, properties, H5P_DEFAULT),
                            "create dataset " + name);
    H5Pclose(properties); H5Sclose(space);
    return dataset;
}

std::vector<hsize_t> DatasetDimensions(hid_t dataset)
{
    hid_t space = H5Valid(H5Dget_space(dataset), "get dataset dimensions");
    const int rank = H5Sget_simple_extent_ndims(space);
    if (rank <= 0)
        throw std::runtime_error("Expected a non-scalar HDF5 dataset");
    std::vector<hsize_t> dimensions(static_cast<std::size_t>(rank));
    H5Check(H5Sget_simple_extent_dims(space, dimensions.data(), nullptr),
            "read dataset dimensions");
    H5Sclose(space);
    return dimensions;
}

void ResizeFirstDimension(hid_t dataset, hsize_t size)
{
    auto dimensions = DatasetDimensions(dataset);
    dimensions[0] = size;
    H5Check(H5Dset_extent(dataset, dimensions.data()), "extend dataset");
}

template <class T>
void WriteSlice(hid_t dataset, const std::vector<hsize_t>& start,
                const std::vector<hsize_t>& count, const T* values)
{
    hid_t fileSpace = H5Valid(H5Dget_space(dataset), "get dataset space");
    H5Check(H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), nullptr,
                                count.data(), nullptr), "select output hyperslab");
    hid_t memorySpace = H5Valid(H5Screate_simple(static_cast<int>(count.size()),
                                                 count.data(), nullptr),
                                "create memory space");
    H5Check(H5Dwrite(dataset, NativeType<T>(), memorySpace, fileSpace, H5P_DEFAULT, values),
            "write dataset slice");
    H5Sclose(memorySpace); H5Sclose(fileSpace);
}

template <class T>
void ReadSlice(hid_t dataset, const std::vector<hsize_t>& start,
               const std::vector<hsize_t>& count, T* values)
{
    hid_t fileSpace = H5Valid(H5Dget_space(dataset), "get dataset space");
    H5Check(H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), nullptr,
                                count.data(), nullptr), "select input hyperslab");
    hid_t memorySpace = H5Valid(H5Screate_simple(static_cast<int>(count.size()),
                                                 count.data(), nullptr),
                                "create input memory space");
    H5Check(H5Dread(dataset, NativeType<T>(), memorySpace, fileSpace, H5P_DEFAULT, values),
            "read dataset slice");
    H5Sclose(memorySpace); H5Sclose(fileSpace);
}

struct CompressedChunk
{
    std::vector<std::uint8_t> bytes;
    std::uint32_t filterMask = 0;
};

class CompressionPool
{
public:
    explicit CompressionPool(std::size_t threadCount)
    {
        workers.reserve(threadCount);
        for (std::size_t i = 0; i < threadCount; ++i)
        {
            workers.emplace_back([this]
            {
                while (true)
                {
                    std::packaged_task<CompressedChunk()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        ready.wait(lock, [this] { return stopping || !tasks.empty(); });
                        if (stopping && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop_front();
                    }
                    task();
                }
            });
        }
    }

    CompressionPool(const CompressionPool&) = delete;
    CompressionPool& operator=(const CompressionPool&) = delete;

    ~CompressionPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& worker : workers)
            worker.join();
    }

    template <class Function>
    std::future<CompressedChunk> Submit(Function&& function)
    {
        std::packaged_task<CompressedChunk()> task(std::forward<Function>(function));
        auto result = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            tasks.emplace_back(std::move(task));
        }
        ready.notify_one();
        return result;
    }

private:
    std::vector<std::thread> workers;
    std::deque<std::packaged_task<CompressedChunk()>> tasks;
    std::mutex mutex;
    std::condition_variable ready;
    bool stopping = false;
};

bool IsLittleEndian()
{
    const std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

template <class T>
std::vector<std::uint8_t> FileOrderBytes(const std::vector<T>& values)
{
    std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
    const auto* source = reinterpret_cast<const std::uint8_t*>(values.data());
    if (IsLittleEndian() || sizeof(T) == 1)
    {
        std::memcpy(bytes.data(), source, bytes.size());
        return bytes;
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        for (std::size_t j = 0; j < sizeof(T); ++j)
            bytes[i * sizeof(T) + j] = source[i * sizeof(T) + sizeof(T) - 1 - j];
    return bytes;
}

template <class T>
CompressedChunk CompressDeflateChunk(std::vector<T> values, unsigned int level)
{
    const std::size_t elementCount = values.size();
    std::vector<std::uint8_t> shuffled(elementCount * sizeof(T));
    const auto* source = reinterpret_cast<const std::uint8_t*>(values.data());
    const bool littleEndian = IsLittleEndian();
    for (std::size_t byte = 0; byte < sizeof(T); ++byte)
    {
        const std::size_t nativeByte = littleEndian ? byte : sizeof(T) - 1 - byte;
        auto* destination = shuffled.data() + byte * elementCount;
        for (std::size_t element = 0; element < elementCount; ++element)
            destination[element] = source[element * sizeof(T) + nativeByte];
    }

    CompressedChunk result;
    if (shuffled.size() > std::numeric_limits<uLong>::max())
        throw std::runtime_error("Compression chunk exceeds zlib's size limit");
    result.bytes.resize(compressBound(static_cast<uLong>(shuffled.size())));
    uLongf compressedSize = static_cast<uLongf>(result.bytes.size());
    const int status = compress2(result.bytes.data(), &compressedSize,
                                    shuffled.data(), static_cast<uLong>(shuffled.size()),
                                    static_cast<int>(level));
    if (status != Z_OK)
        throw std::runtime_error("zlib compression failure: " + std::to_string(status));
    result.bytes.resize(compressedSize);
    if (result.bytes.size() >= shuffled.size())
    {
        result.bytes = FileOrderBytes(values);
        result.filterMask = 0x3U;
    }
    return result;
}

std::optional<unsigned int> DatasetShuffleDeflateLevel(hid_t dataset)
{
    hid_t properties = H5Valid(H5Dget_create_plist(dataset), "get dataset properties");
    const int filterCount = H5Pget_nfilters(properties);
    if (filterCount != 2)
    {
        H5Pclose(properties);
        return std::nullopt;
    }
    unsigned int flags = 0;
    std::size_t valueCount = 0;
    H5Z_filter_t first = H5Pget_filter2(properties, 0, &flags, &valueCount, nullptr,
                                        0, nullptr, nullptr);
    unsigned int level = 0;
    valueCount = 1;
    H5Z_filter_t second = H5Pget_filter2(properties, 1, &flags, &valueCount, &level,
                                         0, nullptr, nullptr);
    H5Pclose(properties);
    if (first != H5Z_FILTER_SHUFFLE || second != H5Z_FILTER_DEFLATE || valueCount != 1)
        return std::nullopt;
    return level;
}

template <class T>
class ParallelDeflateWriter
{
public:
    ParallelDeflateWriter(hid_t dataset, std::size_t firstTuple,
                          unsigned int level, unsigned int threadCount)
        : dataset(dataset), dimensions(DatasetDimensions(dataset)), level(level),
          pool(threadCount), maxPending(std::max<std::size_t>(2, threadCount * 2))
    {
        hid_t properties = H5Valid(H5Dget_create_plist(dataset), "get chunk properties");
        std::vector<hsize_t> chunks(dimensions.size());
        const int rank = H5Pget_chunk(properties, static_cast<int>(chunks.size()), chunks.data());
        H5Pclose(properties);
        if (rank != static_cast<int>(dimensions.size()) || chunks.empty())
            throw std::runtime_error("Parallel compression requires a chunked dataset");
        chunkTuples = static_cast<std::size_t>(chunks[0]);
        elementsPerTuple = 1;
        for (std::size_t dimension = 1; dimension < chunks.size(); ++dimension)
        {
            if (chunks[dimension] != dimensions[dimension])
                throw std::runtime_error("Parallel compression requires complete component chunks");
            elementsPerTuple *= static_cast<std::size_t>(chunks[dimension]);
        }
        currentChunkStart = firstTuple / chunkTuples * chunkTuples;
        expectedTuple = firstTuple;
        current.assign(chunkTuples * elementsPerTuple, T{});
        if (firstTuple != currentChunkStart)
        {
            const std::size_t readable = std::min<std::size_t>(
                chunkTuples, static_cast<std::size_t>(dimensions[0]) - currentChunkStart);
            std::vector<hsize_t> start(dimensions.size(), 0);
            std::vector<hsize_t> count = dimensions;
            start[0] = currentChunkStart;
            count[0] = readable;
            ReadSlice(dataset, start, count, current.data());
        }
    }

    void Write(std::size_t tupleStart, std::size_t tupleCount, const T* values)
    {
        if (tupleStart != expectedTuple)
            throw std::runtime_error("Parallel compression received non-contiguous point data");
        std::size_t consumed = 0;
        while (consumed < tupleCount)
        {
            const std::size_t chunkOffset = expectedTuple - currentChunkStart;
            const std::size_t take = std::min(tupleCount - consumed, chunkTuples - chunkOffset);
            std::copy_n(values + consumed * elementsPerTuple, take * elementsPerTuple,
                        current.data() + chunkOffset * elementsPerTuple);
            consumed += take;
            expectedTuple += take;
            if (expectedTuple == currentChunkStart + chunkTuples)
            {
                SubmitCurrent();
                currentChunkStart += chunkTuples;
                current.assign(chunkTuples * elementsPerTuple, T{});
            }
        }
    }

    void Finish()
    {
        if (expectedTuple > currentChunkStart)
            SubmitCurrent();
        while (!pending.empty())
            WriteOne();
    }

private:
    struct Pending
    {
        std::vector<hsize_t> offset;
        std::future<CompressedChunk> result;
    };

    void SubmitCurrent()
    {
        std::vector<hsize_t> offset(dimensions.size(), 0);
        offset[0] = currentChunkStart;
        auto values = std::move(current);
        pending.push_back({std::move(offset), pool.Submit(
            [values = std::move(values), level = level]() mutable
            {
                return CompressDeflateChunk(std::move(values), level);
            })});
        if (pending.size() >= maxPending)
            WriteOne();
    }

    void WriteOne()
    {
        auto result = pending.front().result.get();
        H5Check(H5Dwrite_chunk(dataset, H5P_DEFAULT, result.filterMask,
                               pending.front().offset.data(), result.bytes.size(),
                               result.bytes.data()),
                "write precompressed dataset chunk");
        pending.pop_front();
    }

    hid_t dataset;
    std::vector<hsize_t> dimensions;
    unsigned int level;
    CompressionPool pool;
    std::size_t maxPending;
    std::size_t chunkTuples = 0;
    std::size_t elementsPerTuple = 0;
    std::size_t currentChunkStart = 0;
    std::size_t expectedTuple = 0;
    std::vector<T> current;
    std::deque<Pending> pending;
};

template <class T>
std::unique_ptr<ParallelDeflateWriter<T>> MakeParallelDeflateWriter(
    hid_t dataset, std::size_t firstTuple = 0)
{
    if (gCompression.threads <= 1)
        return nullptr;
    const auto level = DatasetShuffleDeflateLevel(dataset);
    if (!level)
        return nullptr;
    return std::make_unique<ParallelDeflateWriter<T>>(
        dataset, firstTuple, *level, gCompression.threads);
}

template <class T>
void WriteSmallDataset(hid_t group, const std::string& name, const std::vector<T>& values)
{
    hid_t dataset = CreateDataset<T>(group, name, {values.size()}, std::max<std::size_t>(1, values.size()));
    WriteSlice(dataset, {0}, {values.size()}, values.data());
    H5Dclose(dataset);
}

template <class T>
void WriteResizableDataset(hid_t group, const std::string& name,
                           const std::vector<T>& values)
{
    hid_t dataset = CreateDataset<T>(group, name, {values.size()},
                                     std::max<std::size_t>(1, values.size()), true);
    if (!values.empty())
        WriteSlice(dataset, {0}, {values.size()}, values.data());
    H5Dclose(dataset);
}

template <class T>
std::vector<T> ReadDataset(hid_t parent, const std::string& name)
{
    hid_t dataset = H5Valid(H5Dopen2(parent, name.c_str(), H5P_DEFAULT),
                            "open dataset " + name);
    const auto dimensions = DatasetDimensions(dataset);
    const std::size_t count = std::accumulate(dimensions.begin(), dimensions.end(),
                                               std::size_t{1}, std::multiplies<>());
    std::vector<T> values(count);
    if (count)
        H5Check(H5Dread(dataset, NativeType<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        values.data()), "read dataset " + name);
    H5Dclose(dataset);
    return values;
}

herr_t CollectLinkName(hid_t, const char* name, const H5L_info_t*, void* data)
{
    static_cast<std::vector<std::string>*>(data)->emplace_back(name);
    return 0;
}

std::vector<std::string> ListLinks(hid_t group)
{
    std::vector<std::string> names;
    hsize_t index = 0;
    H5Check(H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, &index,
                        CollectLinkName, &names), "list group links");
    return names;
}

template <class T>
T ReadScalar(adios2::IO& io, adios2::Engine& engine, const std::string& name,
             std::size_t step = 0)
{
    auto variable = io.InquireVariable<T>(name);
    if (!variable)
        throw std::runtime_error("Missing scalar " + name);
    variable.SetStepSelection({step, 1});
    T value{};
    engine.Get(variable, value, adios2::Mode::Sync);
    return value;
}

double ReadTime(adios2::IO& io, adios2::Engine& engine, std::size_t step)
{
    if (io.InquireVariable<double>("time")) return ReadScalar<double>(io, engine, "time", step);
    if (io.InquireVariable<float>("time")) return ReadScalar<float>(io, engine, "time", step);
    throw std::runtime_error("Expected float32 or float64 time");
}

template <class T>
std::vector<typename adios2::Variable<T>::Info>
Blocks(adios2::Engine& engine, adios2::Variable<T>& variable, std::size_t step = 0)
{
    variable.SetStepSelection({step, 1});
    auto blocks = engine.BlocksInfo(variable, step);
    std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) {
        return std::tie(a.WriterID, a.BlockID) < std::tie(b.WriterID, b.BlockID);
    });
    return blocks;
}

template <class T>
void ReadLocalChunk(adios2::Engine& engine, adios2::Variable<T>& variable,
                    const typename adios2::Variable<T>::Info& block, std::size_t step,
                    std::size_t localStart, std::size_t tupleCount, std::vector<T>& values)
{
    adios2::Dims start(block.Count.size(), 0);
    adios2::Dims count = block.Count;
    start[0] = localStart;
    count[0] = tupleCount;
    variable.SetStepSelection({step, 1});
    variable.SetBlockSelection(block.BlockID);
    variable.SetSelection({start, count});
    values.resize(Product(count));
    engine.Get(variable, values.data(), adios2::Mode::Sync);
}

template <class T>
std::size_t PreferredReadTuples(const typename adios2::Variable<T>::Info& block,
                                std::size_t fallbackTuples)
{
    const std::size_t elements = Product(block.Count);
    if (!block.Count.empty() && gFullBlockReadBytes &&
        elements <= gFullBlockReadBytes / sizeof(T))
        return block.Count[0];
    return fallbackTuples;
}

struct MeshLayout
{
    std::string type;
    std::vector<std::size_t> writers;
    std::vector<std::size_t> pointCounts;
    std::vector<std::size_t> pointBases;
    std::vector<std::size_t> cellCounts;
    std::size_t numberOfPoints = 0;
    std::size_t numberOfCells = 0;
};

template <class T>
void ValidateStaticMeshVariable(adios2::Engine& engine, adios2::Variable<T>& variable)
{
    if (variable.Steps() <= 1) return;
    const auto initialBlocks = Blocks(engine, variable);
    for (std::size_t step = 1; step < variable.Steps(); ++step)
    {
        const auto blocks = Blocks(engine, variable, step);
        if (blocks.size() != initialBlocks.size())
            throw std::runtime_error("Only static meshes are supported");
        for (std::size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex)
        {
            const auto& block = blocks[blockIndex];
            const auto& initialBlock = initialBlocks[blockIndex];
            if (block.WriterID != initialBlock.WriterID || block.Count != initialBlock.Count)
                throw std::runtime_error("Only static meshes are supported");
            for (std::size_t start = 0; start < block.Count[0]; start += DEFAULT_READ_CHUNK)
            {
                const auto count = std::min(DEFAULT_READ_CHUNK, block.Count[0] - start);
                std::vector<T> initial, values;
                ReadLocalChunk(engine, variable, initialBlock, 0, start, count, initial);
                ReadLocalChunk(engine, variable, block, step, start, count, values);
                if (values != initial)
                    throw std::runtime_error("Only static meshes are supported");
            }
        }
    }
}

template <class T>
MeshLayout InspectMeshTyped(adios2::IO& io, adios2::Engine& engine)
{
    auto mesh = io.InquireVariable<T>("mesh");
    auto connectivity = io.InquireVariable<std::uint64_t>("connectivity");
    if (!mesh || !connectivity)
        throw std::runtime_error("Input lacks mesh or uint64 connectivity");
    const auto meshBlocks = Blocks(engine, mesh);
    const auto cellBlocks = Blocks(engine, connectivity);
    if (meshBlocks.size() != cellBlocks.size())
        throw std::runtime_error("Mesh and connectivity block counts differ");
    MeshLayout layout;
    layout.type = mesh.Type();
    for (std::size_t i = 0; i < meshBlocks.size(); ++i)
    {
        if (meshBlocks[i].WriterID != cellBlocks[i].WriterID ||
            meshBlocks[i].Count.size() != 2 || meshBlocks[i].Count[1] != 3 ||
            cellBlocks[i].Count.size() != 2 || cellBlocks[i].Count[1] != 9)
            throw std::runtime_error("Unexpected NekRS mesh/connectivity dimensions");
        layout.pointBases.push_back(layout.numberOfPoints);
        layout.writers.push_back(meshBlocks[i].WriterID);
        layout.pointCounts.push_back(meshBlocks[i].Count[0]);
        layout.cellCounts.push_back(cellBlocks[i].Count[0]);
        layout.numberOfPoints += meshBlocks[i].Count[0];
        layout.numberOfCells += cellBlocks[i].Count[0];
    }
    if (!layout.numberOfPoints || !layout.numberOfCells)
        throw std::runtime_error("Input mesh is empty");
    ValidateStaticMeshVariable(engine, mesh);
    ValidateStaticMeshVariable(engine, connectivity);
    return layout;
}

MeshLayout InspectMesh(adios2::IO& io, adios2::Engine& engine)
{
    if (io.InquireVariable<float>("mesh"))
        return InspectMeshTyped<float>(io, engine);
    if (io.InquireVariable<double>("mesh"))
        return InspectMeshTyped<double>(io, engine);
    throw std::runtime_error("Expected float32 or float64 mesh coordinates");
}

struct Field
{
    std::string name;
    std::string type;
    adios2::Dims components;
    std::size_t steps = 1;
};

template <class T>
bool AnalyzePointField(adios2::IO& io, adios2::Engine& engine, const std::string& name,
                       const MeshLayout& mesh, Field& field)
{
    auto variable = io.InquireVariable<T>(name);
    if (!variable)
        return false;
    const auto blocks = Blocks(engine, variable, 0);
    if (blocks.size() != mesh.pointCounts.size())
        return false;
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        if (static_cast<std::size_t>(blocks[i].WriterID) != mesh.writers[i] ||
            blocks[i].Count.empty() || blocks[i].Count.size() > 2 ||
            blocks[i].Count[0] != mesh.pointCounts[i])
            return false;
        adios2::Dims components(blocks[i].Count.begin() + 1, blocks[i].Count.end());
        if (i == 0)
            field.components = components;
        else if (components != field.components)
            return false;
    }
    return true;
}

std::vector<Field> DetectFields(adios2::IO& io, adios2::Engine& engine,
                                const MeshLayout& mesh)
{
    const std::set<std::string> excluded = {"mesh", "connectivity", "globalElementIds",
        "hRefineSchedule", "numOfCells", "numOfPoints", "polynomialOrder", "time", "types"};
    std::vector<Field> fields;
    for (const auto& [name, metadata] : io.AvailableVariables())
    {
        if (excluded.count(name) || metadata.at("SingleValue") == "true")
            continue;
        Field field{name, metadata.at("Type"), {},
                    static_cast<std::size_t>(std::stoull(metadata.at("AvailableStepsCount")))};
        bool valid = false;
        if (field.type == "float")
            valid = AnalyzePointField<float>(io, engine, name, mesh, field);
        else if (field.type == "double")
            valid = AnalyzePointField<double>(io, engine, name, mesh, field);
        if (valid)
            fields.push_back(field);
        else if (field.type == "float" || field.type == "double")
            throw std::runtime_error("Non-scalar variable does not match the point layout: " + name);
    }
    return fields;
}

template <class T>
void ValidateFieldStep(adios2::IO& io, adios2::Engine& engine, const Field& field,
                       const MeshLayout& mesh, std::size_t step)
{
    auto variable = io.InquireVariable<T>(field.name);
    if (variable.Steps() != engine.Steps())
        throw std::runtime_error("Point field must be present at every BP step: " + field.name);
    if (step >= variable.Steps())
        throw std::runtime_error("Requested source step is out of range for " + field.name);
    const auto blocks = Blocks(engine, variable, step);
    if (blocks.size() != mesh.pointCounts.size())
        throw std::runtime_error("Writer blocks changed for " + field.name);
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        adios2::Dims expected = {mesh.pointCounts[i]};
        expected.insert(expected.end(), field.components.begin(), field.components.end());
        if (static_cast<std::size_t>(blocks[i].WriterID) != mesh.writers[i] || blocks[i].Count != expected)
            throw std::runtime_error("Writer/shape mismatch for " + field.name +
                                     " at step " + std::to_string(step));
    }
}

template <class T>
void WritePointFieldPositions(adios2::IO& io, adios2::Engine& engine,
                              hid_t pointData, hid_t pointDataOffsets,
                              const Field& field, const MeshLayout& mesh,
                              const std::vector<std::size_t>& sourceSteps,
                              const std::vector<std::size_t>& positions,
                              std::size_t totalPositions, std::size_t chunkTuples)
{
    const auto fieldStarted = std::chrono::steady_clock::now();
    double readSeconds = 0.0;
    double outputSeconds = 0.0;
    std::size_t readCalls = 0;
    auto variable = io.InquireVariable<T>(field.name);
    if (!variable)
        throw std::runtime_error("Input no longer contains point field " + field.name);

    const bool create = !LinkExists(pointData, field.name);
    hid_t dataset = -1;
    if (create)
    {
        std::vector<hsize_t> dimensions = {totalPositions * mesh.numberOfPoints};
        dimensions.insert(dimensions.end(), field.components.begin(), field.components.end());
        dataset = CreateDataset<T>(pointData, field.name, dimensions,
                                   gCompression.chunkTuples, true);
    }
    else
    {
        dataset = H5Valid(H5Dopen2(pointData, field.name.c_str(), H5P_DEFAULT),
                          "open point field " + field.name);
        const auto dimensions = DatasetDimensions(dataset);
        if (dimensions.size() != field.components.size() + 1 ||
            !std::equal(field.components.begin(), field.components.end(),
                        dimensions.begin() + 1))
            throw std::runtime_error("Existing VTKHDF component shape differs for " + field.name);
        const H5T_class_t expectedClass = std::is_floating_point_v<T> ? H5T_FLOAT : H5T_INTEGER;
        hid_t type = H5Valid(H5Dget_type(dataset), "get point-field type");
        const bool typeMatches = H5Tget_class(type) == expectedClass && H5Tget_size(type) == sizeof(T);
        H5Tclose(type);
        if (!typeMatches)
            throw std::runtime_error("Existing VTKHDF type differs for " + field.name);
        ResizeFirstDimension(dataset, totalPositions * mesh.numberOfPoints);
    }

    hid_t offsets = -1;
    if (LinkExists(pointDataOffsets, field.name))
    {
        offsets = H5Valid(H5Dopen2(pointDataOffsets, field.name.c_str(), H5P_DEFAULT),
                          "open point-data offsets " + field.name);
        ResizeFirstDimension(offsets, totalPositions);
    }
    else
    {
        offsets = CreateDataset<std::int64_t>(pointDataOffsets, field.name,
                                              {totalPositions},
                                              std::max<std::size_t>(1, totalPositions), true);
    }

    std::unique_ptr<ParallelDeflateWriter<T>> parallelWriter;
    if (!positions.empty())
        parallelWriter = MakeParallelDeflateWriter<T>(
            dataset, positions.front() * mesh.numberOfPoints);

    for (std::size_t item = 0; item < positions.size(); ++item)
    {
        const std::size_t sourceStep = sourceSteps[item];
        const std::size_t position = positions[item];
        if (sourceStep >= field.steps)
            throw std::runtime_error("Field " + field.name + " has no requested source step " +
                                     std::to_string(sourceStep));
        const std::int64_t offset = static_cast<std::int64_t>(position * mesh.numberOfPoints);
        WriteSlice(offsets, {position}, {1}, &offset);
        const auto blocks = Blocks(engine, variable, sourceStep);
        if (blocks.size() != mesh.pointCounts.size())
            throw std::runtime_error("Point-field block count changed for " + field.name);
        for (std::size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex)
        {
            const std::size_t readTuples = PreferredReadTuples<T>(
                blocks[blockIndex], chunkTuples);
            for (std::size_t localStart = 0; localStart < mesh.pointCounts[blockIndex];
                 localStart += readTuples)
            {
                const std::size_t count = std::min(readTuples,
                    mesh.pointCounts[blockIndex] - localStart);
                std::vector<T> values;
                const auto readStarted = std::chrono::steady_clock::now();
                ReadLocalChunk(engine, variable, blocks[blockIndex], sourceStep,
                               localStart, count, values);
                ++readCalls;
                readSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - readStarted).count();
                std::vector<hsize_t> start = {
                    position * mesh.numberOfPoints + mesh.pointBases[blockIndex] + localStart};
                std::vector<hsize_t> writeCount = {count};
                start.insert(start.end(), field.components.size(), 0);
                writeCount.insert(writeCount.end(), field.components.begin(), field.components.end());
                const auto outputStarted = std::chrono::steady_clock::now();
                if (parallelWriter)
                    parallelWriter->Write(static_cast<std::size_t>(start[0]), count,
                                          values.data());
                else
                    WriteSlice(dataset, start, writeCount, values.data());
                outputSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - outputStarted).count();
            }
        }
        std::cout << "  " << field.name << ": source step " << sourceStep
                  << " -> VTK step " << position << " complete\n";
    }
    if (parallelWriter)
    {
        const auto outputStarted = std::chrono::steady_clock::now();
        parallelWriter->Finish();
        outputSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - outputStarted).count();
    }
    const double totalSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fieldStarted).count();
    std::cout << "    timing: total " << totalSeconds << " s, ADIOS read "
              << readSeconds << " s (" << readCalls << " Get calls), output pipeline "
              << outputSeconds << " s\n";
    H5Dclose(offsets);
    H5Dclose(dataset);
}

template <class T>
void WriteGeometry(adios2::IO& io, adios2::Engine& engine, hid_t root,
                   const MeshLayout& layout, std::size_t chunkTuples)
{
    const auto geometryStarted = std::chrono::steady_clock::now();
    const auto pointsStarted = geometryStarted;
    std::size_t pointsGetCalls = 0;
    std::size_t connectivityGetCalls = 0;
    auto mesh = io.InquireVariable<T>("mesh");
    const auto meshBlocks = Blocks(engine, mesh);
    hid_t points = CreateDataset<T>(root, "Points", {layout.numberOfPoints, 3});
    auto pointsWriter = MakeParallelDeflateWriter<T>(points);
    for (std::size_t blockIndex = 0; blockIndex < meshBlocks.size(); ++blockIndex)
    {
        const std::size_t readTuples = PreferredReadTuples<T>(
            meshBlocks[blockIndex], chunkTuples);
        for (std::size_t localStart = 0; localStart < layout.pointCounts[blockIndex];
             localStart += readTuples)
        {
            const std::size_t count = std::min(readTuples,
                layout.pointCounts[blockIndex] - localStart);
            std::vector<T> values;
            ReadLocalChunk(engine, mesh, meshBlocks[blockIndex], 0, localStart, count, values);
            ++pointsGetCalls;
            if (pointsWriter)
                pointsWriter->Write(layout.pointBases[blockIndex] + localStart, count,
                                    values.data());
            else
                WriteSlice(points, {layout.pointBases[blockIndex] + localStart, 0}, {count, 3},
                           values.data());
        }
    }
    if (pointsWriter)
        pointsWriter->Finish();
    H5Dclose(points);
    const auto connectivityStarted = std::chrono::steady_clock::now();
    const double pointsSeconds = std::chrono::duration<double>(
        connectivityStarted - pointsStarted).count();

    auto connectivity = io.InquireVariable<std::uint64_t>("connectivity");
    const auto cellBlocks = Blocks(engine, connectivity);
    hid_t idsDataset = CreateDataset<std::int64_t>(root, "Connectivity",
        {layout.numberOfCells * 8}, gCompression.chunkTuples * 8);
    auto idsWriter = MakeParallelDeflateWriter<std::int64_t>(idsDataset);
    std::size_t globalCellBase = 0;
    for (std::size_t blockIndex = 0; blockIndex < cellBlocks.size(); ++blockIndex)
    {
        const std::size_t readTuples = PreferredReadTuples<std::uint64_t>(
            cellBlocks[blockIndex], chunkTuples);
        for (std::size_t localStart = 0; localStart < layout.cellCounts[blockIndex];
             localStart += readTuples)
        {
            const std::size_t count = std::min(readTuples,
                layout.cellCounts[blockIndex] - localStart);
            std::vector<std::uint64_t> source;
            ReadLocalChunk(engine, connectivity, cellBlocks[blockIndex], 0,
                           localStart, count, source);
            ++connectivityGetCalls;
            const std::size_t conversionTuples = std::min(
                gCompression.chunkTuples, DEFAULT_HDF_CHUNK_TUPLES);
            for (std::size_t sourceStart = 0; sourceStart < count;
                 sourceStart += conversionTuples)
            {
                const std::size_t convertedCount = std::min(
                    conversionTuples, count - sourceStart);
                std::vector<std::int64_t> ids(convertedCount * 8);
                for (std::size_t row = 0; row < convertedCount; ++row)
                {
                    const std::size_t sourceRow = sourceStart + row;
                    if (source[sourceRow * 9] != 8)
                        throw std::runtime_error("Non-hexahedral connectivity encountered");
                    for (std::size_t component = 0; component < 8; ++component)
                    {
                        const auto localId = source[sourceRow * 9 + component + 1];
                        if (localId >= layout.pointCounts[blockIndex])
                            throw std::runtime_error(
                                "Connectivity references an invalid local point");
                        ids[row * 8 + component] = static_cast<std::int64_t>(
                            localId + layout.pointBases[blockIndex]);
                    }
                }
                const std::size_t outputStart =
                    (globalCellBase + localStart + sourceStart) * 8;
                if (idsWriter)
                    idsWriter->Write(outputStart, convertedCount * 8, ids.data());
                else
                    WriteSlice(idsDataset, {outputStart}, {convertedCount * 8}, ids.data());
            }
        }
        globalCellBase += layout.cellCounts[blockIndex];
    }
    if (idsWriter)
        idsWriter->Finish();
    H5Dclose(idsDataset);
    const auto offsetsStarted = std::chrono::steady_clock::now();
    const double connectivitySeconds = std::chrono::duration<double>(
        offsetsStarted - connectivityStarted).count();

    hid_t offsets = CreateDataset<std::int64_t>(root, "Offsets", {layout.numberOfCells + 1});
    auto offsetsWriter = MakeParallelDeflateWriter<std::int64_t>(offsets);
    for (std::size_t start = 0; start < layout.numberOfCells + 1; start += chunkTuples)
    {
        const std::size_t count = std::min(chunkTuples, layout.numberOfCells + 1 - start);
        std::vector<std::int64_t> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = static_cast<std::int64_t>((start + i) * 8);
        if (offsetsWriter)
            offsetsWriter->Write(start, count, values.data());
        else
            WriteSlice(offsets, {start}, {count}, values.data());
    }
    if (offsetsWriter)
        offsetsWriter->Finish();
    H5Dclose(offsets);
    const auto typesStarted = std::chrono::steady_clock::now();
    const double offsetsSeconds = std::chrono::duration<double>(
        typesStarted - offsetsStarted).count();

    hid_t types = CreateDataset<std::uint8_t>(root, "Types", {layout.numberOfCells});
    auto typesWriter = MakeParallelDeflateWriter<std::uint8_t>(types);
    std::vector<std::uint8_t> typeBuffer(std::min(chunkTuples, layout.numberOfCells), VTK_HEXAHEDRON);
    for (std::size_t start = 0; start < layout.numberOfCells; start += chunkTuples)
    {
        const std::size_t count = std::min(chunkTuples, layout.numberOfCells - start);
        if (typesWriter)
            typesWriter->Write(start, count, typeBuffer.data());
        else
            WriteSlice(types, {start}, {count}, typeBuffer.data());
    }
    if (typesWriter)
        typesWriter->Finish();
    H5Dclose(types);
    const auto geometryFinished = std::chrono::steady_clock::now();
    const double typesSeconds = std::chrono::duration<double>(
        geometryFinished - typesStarted).count();
    const double totalSeconds = std::chrono::duration<double>(
        geometryFinished - geometryStarted).count();
    std::cout << "Geometry timing: total " << totalSeconds << " s (points "
              << pointsSeconds << " s/" << pointsGetCalls << " Get, connectivity "
              << connectivitySeconds << " s/" << connectivityGetCalls << " Get, offsets "
              << offsetsSeconds << " s, types "
              << typesSeconds << " s)\n";
}

std::size_t WriteStaticFieldData(adios2::IO& io, adios2::Engine& engine,
                                 hid_t fieldData, const MeshLayout& mesh,
                                 std::size_t order, std::size_t chunkTuples)
{
    WriteSmallDataset<std::uint32_t>(fieldData, "polynomialOrder",
                                     {static_cast<std::uint32_t>(order)});
    auto idsVariable = io.InquireVariable<std::uint64_t>("globalElementIds");
    if (!idsVariable)
        return 0;
    const auto blocks = Blocks(engine, idsVariable);
    std::size_t totalIds = 0;
    if (blocks.size() != mesh.writers.size())
        throw std::runtime_error("globalElementIds writer count does not match the mesh");
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        if (blocks[i].Count.size() != 1 ||
            static_cast<std::size_t>(blocks[i].WriterID) != mesh.writers[i])
            throw std::runtime_error("Invalid globalElementIds writer/shape");
        totalIds += blocks[i].Count[0];
    }
    if (!totalIds) return 0;
    hid_t ids = CreateDataset<std::uint64_t>(fieldData, "globalElementIds", {totalIds});
    std::size_t base = 0;
    for (const auto& block : blocks)
    {
        for (std::size_t localStart = 0; localStart < block.Count[0]; localStart += chunkTuples)
        {
            const std::size_t count = std::min(chunkTuples, block.Count[0] - localStart);
            std::vector<std::uint64_t> values;
            ReadLocalChunk(engine, idsVariable, block, 0, localStart, count, values);
            WriteSlice(ids, {base + localStart}, {count}, values.data());
        }
        base += block.Count[0];
    }
    H5Dclose(ids);

    return totalIds;
}

void UpdateFieldDataStepMetadata(hid_t fieldOffsets, hid_t fieldSizes,
                                 std::size_t oldCount, std::size_t newCount,
                                 std::size_t globalIdCount)
{
    for (const auto& name : {"polynomialOrder", "globalElementIds"})
    {
        if (name == std::string("globalElementIds") && globalIdCount == 0)
            continue;
        hid_t offsets = -1;
        if (LinkExists(fieldOffsets, name))
        {
            offsets = H5Valid(H5Dopen2(fieldOffsets, name, H5P_DEFAULT),
                              std::string("open field offsets ") + name);
            ResizeFirstDimension(offsets, newCount);
        }
        else
        {
            offsets = CreateDataset<std::int64_t>(fieldOffsets, name, {newCount},
                                                  std::max<std::size_t>(1, newCount), true);
        }
        std::vector<std::int64_t> zeros(newCount - oldCount, 0);
        if (!zeros.empty())
            WriteSlice(offsets, {oldCount}, {zeros.size()}, zeros.data());
        H5Dclose(offsets);

        hid_t sizes = -1;
        if (LinkExists(fieldSizes, name))
        {
            sizes = H5Valid(H5Dopen2(fieldSizes, name, H5P_DEFAULT),
                            std::string("open field sizes ") + name);
            ResizeFirstDimension(sizes, newCount);
        }
        else
        {
            sizes = CreateDataset<std::int64_t>(fieldSizes, name, {newCount, 2},
                                                std::max<std::size_t>(1, newCount), true);
        }
        std::vector<std::int64_t> values((newCount - oldCount) * 2);
        const std::int64_t tuples = name == std::string("polynomialOrder")
            ? 1 : static_cast<std::int64_t>(globalIdCount);
        for (std::size_t i = 0; i < newCount - oldCount; ++i)
        {
            values[i * 2] = 1;
            values[i * 2 + 1] = tuples;
        }
        if (!values.empty())
            WriteSlice(sizes, {oldCount, 0}, {newCount - oldCount, 2}, values.data());
        H5Dclose(sizes);
    }
}

struct Options
{
    fs::path input;
    fs::path output;
    std::optional<std::vector<std::string>> fields;
    std::optional<std::vector<std::size_t>> steps;
    std::size_t chunkTuples = DEFAULT_READ_CHUNK;
    std::size_t hdfChunkTuples = DEFAULT_HDF_CHUNK_TUPLES;
    std::size_t fullBlockReadLimitMiB = DEFAULT_FULL_BLOCK_READ_LIMIT_MIB;
    bool compression = true;
    unsigned int compressionLevel = 1;
    unsigned int compressionThreads = 4;
    bool overwrite = false;
};

std::size_t ParseUnsigned(const std::string& text)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected a non-negative integer: " + text);
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Integer is too large: " + text);
    return static_cast<std::size_t>(value);
}

std::vector<std::string> Split(const std::string& text, char delimiter)
{
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string value;
    while (std::getline(stream, value, delimiter))
    {
        if (!value.empty())
            values.push_back(value);
    }
    return values;
}

Options ParseOptions(int argc, char** argv)
{
    if (argc < 3)
        throw std::invalid_argument("missing INPUT.bp and OUTPUT.vtkhdf");
    Options options;
    options.input = fs::absolute(argv[1]);
    options.output = fs::absolute(argv[2]);
    for (int index = 3; index < argc; ++index)
    {
        const std::string argument = argv[index];
        auto nextValue = [&]() -> std::string {
            if (++index >= argc)
                throw std::invalid_argument("missing value after " + argument);
            return argv[index];
        };
        if (argument == "--fields")
        {
            options.fields = Split(nextValue(), ',');
            if (options.fields->empty())
                throw std::invalid_argument("--fields cannot be empty");
        }
        else if (argument == "--steps")
        {
            std::vector<std::size_t> steps;
            for (const auto& value : Split(nextValue(), ','))
                steps.push_back(ParseUnsigned(value));
            if (steps.empty())
                throw std::invalid_argument("--steps cannot be empty");
            std::sort(steps.begin(), steps.end());
            steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
            options.steps = std::move(steps);
        }
        else if (argument == "--chunk-tuples")
        {
            options.chunkTuples = ParseUnsigned(nextValue());
            if (!options.chunkTuples)
                throw std::invalid_argument("--chunk-tuples must be positive");
        }
        else if (argument == "--hdf-chunk-tuples")
        {
            options.hdfChunkTuples = ParseUnsigned(nextValue());
            if (!options.hdfChunkTuples || options.hdfChunkTuples > 10'000'000)
                throw std::invalid_argument("--hdf-chunk-tuples must be in [1, 10000000]");
        }
        else if (argument == "--full-block-read-limit-mib")
        {
            options.fullBlockReadLimitMiB = ParseUnsigned(nextValue());
            if (options.fullBlockReadLimitMiB > 65536)
                throw std::invalid_argument(
                    "--full-block-read-limit-mib must be in [0, 65536]");
        }
        else if (argument == "--compression")
        {
            const std::string value = nextValue();
            if (value == "gzip" || value == "deflate")
                options.compression = true;
            else if (value == "none")
                options.compression = false;
            else
                throw std::invalid_argument("--compression must be gzip or none");
        }
        else if (argument == "--compression-level")
        {
            const auto value = ParseUnsigned(nextValue());
            if (value < 1 || value > 9)
                throw std::invalid_argument("--compression-level must be in [1, 9]");
            options.compressionLevel = static_cast<unsigned int>(value);
        }
        else if (argument == "--compression-threads")
        {
            const auto value = ParseUnsigned(nextValue());
            if (value < 1 || value > 64)
                throw std::invalid_argument("--compression-threads must be in [1, 64]");
            options.compressionThreads = static_cast<unsigned int>(value);
        }
        else if (argument == "--overwrite")
            options.overwrite = true;
        else if (argument == "--help" || argument == "-h")
            throw std::invalid_argument("");
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
    return options;
}

void PrintUsage(const char* executable)
{
    std::cerr
        << "Usage: " << executable << " INPUT.bp OUTPUT.vtkhdf [options]\n"
        << "Options:\n"
        << "  --fields NAME1,NAME2   Fields to add (default: all point fields)\n"
        << "  --steps 0,1,2          Source BP steps to add (default: all common steps)\n"
        << "  --chunk-tuples N       ADIOS2 read block size (default: "
        << DEFAULT_READ_CHUNK << ")\n"
        << "  --hdf-chunk-tuples N   HDF5 chunk tuples (default: "
        << DEFAULT_HDF_CHUNK_TUPLES << ")\n"
        << "  --full-block-read-limit-mib N  max raw ADIOS block read (default: "
        << DEFAULT_FULL_BLOCK_READ_LIMIT_MIB << " MiB; 0 disables)\n"
        << "  --compression MODE     gzip or none (default: gzip)\n"
        << "  --compression-level N  gzip level 1-9 (default: 1)\n"
        << "  --compression-threads N  parallel gzip workers (default: 4)\n"
        << "  --overwrite            Replace an existing output instead of appending\n";
}

void ConfigureCompression(const Options& options)
{
    gCompression.enabled = options.compression;
    gCompression.level = options.compressionLevel;
    gCompression.threads = options.compressionThreads;
    gCompression.chunkTuples = options.hdfChunkTuples;
    if (options.fullBlockReadLimitMiB >
        std::numeric_limits<std::size_t>::max() / (std::size_t{1} << 20))
        throw std::invalid_argument("--full-block-read-limit-mib is too large");
    gFullBlockReadBytes = options.fullBlockReadLimitMiB << 20;
    if (!gCompression.enabled)
        return;
    if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0)
        throw std::runtime_error(
            "HDF5 DEFLATE filter is unavailable; rebuild HDF5 with zlib or use --compression none");
    unsigned int configuration = 0;
    H5Check(H5Zget_filter_info(H5Z_FILTER_DEFLATE, &configuration),
            "query HDF5 DEFLATE filter");
    if ((configuration & H5Z_FILTER_CONFIG_ENCODE_ENABLED) == 0)
        throw std::runtime_error(
            "HDF5 DEFLATE encoder is unavailable; use --compression none");
}

std::map<std::string, Field> IndexFields(const std::vector<Field>& fields)
{
    std::map<std::string, Field> result;
    for (const auto& field : fields)
        result.emplace(field.name, field);
    return result;
}

std::vector<std::string> RequestedFieldNames(
    const Options& options, const std::map<std::string, Field>& available)
{
    std::vector<std::string> names;
    if (options.fields)
        names = *options.fields;
    else
        for (const auto& [name, field] : available)
        {
            (void)field;
            names.push_back(name);
        }
    for (const auto& name : names)
    {
        if (!available.count(name))
            throw std::runtime_error("Requested point field is unavailable: " + name);
        if (name.find('/') != std::string::npos || name.find('.') != std::string::npos)
            throw std::runtime_error("VTKHDF field names cannot contain '/' or '.': " + name);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

std::vector<std::size_t> RequestedSteps(
    const Options& options, const std::vector<std::string>& fieldNames,
    const std::map<std::string, Field>& available)
{
    if (options.steps)
        return *options.steps;
    std::size_t count = std::numeric_limits<std::size_t>::max();
    for (const auto& name : fieldNames)
    {
        const auto iterator = available.find(name);
        if (iterator == available.end())
            throw std::runtime_error("Cannot determine source steps for field " + name);
        count = std::min(count, iterator->second.steps);
    }
    if (count == 0 || count == std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("No common source time steps are available");
    std::vector<std::size_t> steps(count);
    std::iota(steps.begin(), steps.end(), 0);
    return steps;
}

template <class T>
void ExtendAndWrite(hid_t group, const std::string& name, std::size_t oldCount,
                    std::size_t newCount, const std::vector<T>& appended)
{
    hid_t dataset = -1;
    if (LinkExists(group, name))
    {
        dataset = H5Valid(H5Dopen2(group, name.c_str(), H5P_DEFAULT),
                          "open temporal dataset " + name);
        ResizeFirstDimension(dataset, newCount);
    }
    else
    {
        dataset = CreateDataset<T>(group, name, {newCount},
                                   std::max<std::size_t>(1, newCount), true);
    }
    if (!appended.empty())
        WriteSlice(dataset, {oldCount}, {appended.size()}, appended.data());
    H5Dclose(dataset);
}

void UpdateStepMetadata(adios2::IO& io, adios2::Engine& engine,
                        hid_t stepsGroup, hid_t stateGroup,
                        const std::vector<std::size_t>& allSourceSteps,
                        std::size_t oldCount)
{
    const std::size_t newCount = allSourceSteps.size();
    std::vector<double> times;
    std::vector<std::int64_t> indices;
    times.reserve(newCount - oldCount);
    indices.reserve(newCount - oldCount);
    for (std::size_t position = oldCount; position < newCount; ++position)
    {
        times.push_back(ReadTime(io, engine, allSourceSteps[position]));
        indices.push_back(static_cast<std::int64_t>(allSourceSteps[position]));
    }
    ExtendAndWrite(stepsGroup, "Values", oldCount, newCount, times);
    std::vector<std::int64_t> zeros(newCount - oldCount, 0);
    std::vector<std::int64_t> ones(newCount - oldCount, 1);
    for (const auto& name : {"PointOffsets", "PartOffsets"})
        ExtendAndWrite(stepsGroup, name, oldCount, newCount, zeros);
    for (const auto& name : {"CellOffsets", "ConnectivityIdOffsets"})
    {
        hid_t dataset = LinkExists(stepsGroup, name)
            ? H5Valid(H5Dopen2(stepsGroup, name, H5P_DEFAULT), "open topology offsets")
            : CreateDataset<std::int64_t>(stepsGroup, name, {newCount, 1}, 1, true);
        ResizeFirstDimension(dataset, newCount);
        if (!zeros.empty())
            WriteSlice(dataset, {oldCount, 0}, {zeros.size(), 1}, zeros.data());
        H5Dclose(dataset);
    }
    ExtendAndWrite(stepsGroup, "NumberOfParts", oldCount, newCount, ones);
    ExtendAndWrite(stateGroup, "StepIndices", oldCount, newCount, indices);
}

std::int64_t ReadFirstInt64(hid_t group, const std::string& name)
{
    const auto values = ReadDataset<std::int64_t>(group, name);
    if (values.empty())
        throw std::runtime_error("Empty dataset " + name);
    return values.front();
}

struct ScalarMetadata
{
    std::string name, type;
    std::size_t steps;
};

template <class F>
void DispatchScalar(const ScalarMetadata& scalar, F&& function)
{
    if (scalar.type == "float") function(float{});
    else if (scalar.type == "double") function(double{});
    else if (scalar.type == "int32_t") function(std::int32_t{});
    else if (scalar.type == "int64_t") function(std::int64_t{});
    else if (scalar.type == "uint32_t") function(std::uint32_t{});
    else if (scalar.type == "uint64_t") function(std::uint64_t{});
    else throw std::runtime_error("Unsupported scalar type for " + scalar.name + ": " + scalar.type);
}

std::vector<ScalarMetadata> DetectScalars(adios2::IO& io, adios2::Engine& engine)
{
    const std::set<std::string> excluded = {
        "numOfCells", "numOfPoints", "types", "time", "polynomialOrder"};
    std::vector<ScalarMetadata> result;
    for (const auto& [name, info] : io.AvailableVariables())
    {
        if (excluded.count(name) || info.at("SingleValue") != "true") continue;
        if (name.find('/') != std::string::npos)
            throw std::runtime_error("Metadata names cannot contain '/': " + name);
        ScalarMetadata scalar{name, info.at("Type"), std::stoull(info.at("AvailableStepsCount"))};
        if (scalar.steps != 1 && scalar.steps != engine.Steps())
            throw std::runtime_error("Scalar must be static or present at every step: " + name);
        result.push_back(scalar);
    }
    return result;
}

template <class T>
std::vector<T> ScalarValues(adios2::IO& io, adios2::Engine& engine,
                            const ScalarMetadata& scalar,
                            const std::vector<std::size_t>& sourceSteps)
{
    auto variable = io.InquireVariable<T>(scalar.name);
    if (engine.BlocksInfo(variable, 0).empty())
        throw std::runtime_error("Scalar is missing at the first step: " + scalar.name);
    std::vector<T> values;
    for (const auto step : sourceSteps)
        values.push_back(ReadScalar<T>(io, engine, scalar.name, scalar.steps == 1 ? 0 : step));
    return values;
}

void WriteScalarMetadata(adios2::IO& io, adios2::Engine& engine,
                         hid_t fieldData, hid_t offsets, hid_t sizes,
                         const std::vector<ScalarMetadata>& scalars,
                         const std::vector<std::size_t>& sourceSteps)
{
    const std::size_t count = sourceSteps.size();
    for (const auto& scalar : scalars)
    {
        DispatchScalar(scalar, [&](auto value) {
            using T = decltype(value);
            const auto values = ScalarValues<T>(io, engine, scalar, sourceSteps);
            hid_t dataset = LinkExists(fieldData, scalar.name)
                ? H5Valid(H5Dopen2(fieldData, scalar.name.c_str(), H5P_DEFAULT), "open scalar data")
                : CreateDataset<T>(fieldData, scalar.name, {count}, 1, true);
            ResizeFirstDimension(dataset, count);
            WriteSlice(dataset, {0}, {count}, values.data());
            H5Dclose(dataset);
        });
        std::vector<std::int64_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        ExtendAndWrite(offsets, scalar.name, 0, count, indices);
        hid_t dataset = LinkExists(sizes, scalar.name)
            ? H5Valid(H5Dopen2(sizes, scalar.name.c_str(), H5P_DEFAULT), "open scalar sizes")
            : CreateDataset<std::int64_t>(sizes, scalar.name, {count, 2}, 1, true);
        ResizeFirstDimension(dataset, count);
        std::vector<std::int64_t> ones(count * 2, 1);
        WriteSlice(dataset, {0, 0}, {count, 2}, ones.data());
        H5Dclose(dataset);
    }
}

void ValidateExistingFile(hid_t file, const MeshLayout& mesh, std::size_t order,
                           const fs::path& input)
{
    if (!LinkExists(file, "VTKHDF") || !LinkExists(file, STATE_GROUP))
        throw std::runtime_error("Appending requires a VTKHDF file created by bp2vtkhdf");
    hid_t state = OpenGroup(file, STATE_GROUP);
    if (!H5Aexists(state, "NativeSchemaVersion") || !LinkExists(state, "StepIndices"))
    {
        H5Gclose(state);
        throw std::runtime_error("Missing bp2vtkhdf append metadata");
    }
    std::int64_t version = 0, inProgress = 0;
    hid_t attr = H5Valid(H5Aopen(state, "NativeSchemaVersion", H5P_DEFAULT), "open schema version");
    H5Check(H5Aread(attr, H5T_NATIVE_INT64, &version), "read schema version");
    H5Aclose(attr);
    if (version != 2)
        throw std::runtime_error("bp2vtkhdf format version " + std::to_string(version) + "; required version: 2");
    attr = H5Valid(H5Aopen(state, "InProgress", H5P_DEFAULT), "open completion flag");
    H5Check(H5Aread(attr, H5T_NATIVE_INT64, &inProgress), "read completion flag");
    H5Aclose(attr);
    if (inProgress)
        throw std::runtime_error("Previous write did not finish; recreate the output with --overwrite");
    if (fs::weakly_canonical(ReadStringAttribute(state, "SourceBP")) != fs::canonical(input))
        throw std::runtime_error("Existing output belongs to a different BP input");
    const auto counts = ReadDataset<std::uint64_t>(state, "PointCounts");
    const auto writers = ReadDataset<std::uint64_t>(state, "WriterIds");
    if (counts.size() != mesh.pointCounts.size() || writers.size() != mesh.writers.size() ||
        !std::equal(counts.begin(), counts.end(), mesh.pointCounts.begin()) ||
        !std::equal(writers.begin(), writers.end(), mesh.writers.begin()))
        throw std::runtime_error("Writer partitioning changed since the output was created");
    H5Gclose(state);
    hid_t root = OpenGroup(file, "VTKHDF");
    if (ReadFirstInt64(root, "NumberOfPoints") != static_cast<std::int64_t>(mesh.numberOfPoints) ||
        ReadFirstInt64(root, "NumberOfCells") != static_cast<std::int64_t>(mesh.numberOfCells))
    {
        H5Gclose(root);
        throw std::runtime_error("Existing output mesh size differs from the BP input");
    }
    hid_t fieldData = OpenGroup(root, "FieldData");
    const auto storedOrder = ReadDataset<std::uint32_t>(fieldData, "polynomialOrder");
    H5Gclose(fieldData);
    H5Gclose(root);
    if (storedOrder.size() != 1 || storedOrder[0] != order)
        throw std::runtime_error("Existing output polynomial order differs from the BP input");
}
}

int main(int argc, char** argv)
{
    bool createdOutput = false;
    fs::path writePath;
    hid_t file = -1;
    try
    {
        if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
        {
            PrintUsage(argv[0]);
            return 0;
        }
        const Options options = ParseOptions(argc, argv);
        ConfigureCompression(options);
        if (!fs::exists(options.input))
            throw std::runtime_error("Input does not exist: " + options.input.string());
        if (fs::weakly_canonical(options.output) == fs::canonical(options.input) ||
            (fs::exists(options.output) && fs::equivalent(options.input, options.output)))
            throw std::runtime_error("Input and output must be different paths");
        const bool outputExists = fs::exists(options.output) && !options.overwrite;
        const auto started = std::chrono::steady_clock::now();
        adios2::ADIOS adios;
        auto io = adios.DeclareIO("Input");
        io.SetEngine("BPFile");
        auto engine = io.Open(options.input.string(), adios2::Mode::ReadRandomAccess);
        const MeshLayout mesh = InspectMesh(io, engine);
        const auto availableFields = IndexFields(DetectFields(io, engine, mesh));
        if (availableFields.empty()) throw std::runtime_error("No point fields detected");
        const auto order = ReadScalar<std::uint32_t>(io, engine, "polynomialOrder");
        if (ReadScalar<std::uint32_t>(io, engine, "types") != VTK_HEXAHEDRON)
            throw std::runtime_error("Only VTK_HEXAHEDRON type 12 is supported");
        const auto scalars = DetectScalars(io, engine);

        hid_t access = H5Valid(H5Pcreate(H5P_FILE_ACCESS), "create file access properties");
        H5Check(H5Pset_fclose_degree(access, H5F_CLOSE_STRONG), "set file close policy");
        std::vector<std::int64_t> storedIndices;
        std::vector<std::string> existingFields;
        std::size_t globalIdCount = 0;
        if (outputExists)
        {
            file = H5Valid(H5Fopen(options.output.string().c_str(), H5F_ACC_RDONLY, access),
                           "inspect existing output");
            ValidateExistingFile(file, mesh, order, options.input);
            hid_t state = OpenGroup(file, STATE_GROUP);
            storedIndices = ReadDataset<std::int64_t>(state, "StepIndices");
            H5Gclose(state);
            hid_t pointData = OpenGroup(file, "VTKHDF/PointData");
            existingFields = ListLinks(pointData);
            for (const auto& name : existingFields)
            {
                const auto it = availableFields.find(name);
                if (it == availableFields.end())
                    throw std::runtime_error("Input lacks existing field " + name);
                hid_t dataset = H5Valid(H5Dopen2(pointData, name.c_str(), H5P_DEFAULT), "inspect field");
                std::vector<hsize_t> expected = {storedIndices.size() * mesh.numberOfPoints};
                expected.insert(expected.end(), it->second.components.begin(), it->second.components.end());
                if (DatasetDimensions(dataset) != expected)
                    throw std::runtime_error("Existing field dimensions differ: " + name);
                hid_t type = H5Valid(H5Dget_type(dataset), "inspect field type");
                const bool matches = H5Tequal(type, it->second.type == "float" ? FileType<float>() : FileType<double>()) > 0;
                H5Tclose(type); H5Dclose(dataset);
                if (!matches) throw std::runtime_error("Existing field type differs: " + name);
            }
            H5Gclose(pointData);
            const auto times = ReadDataset<double>(file, "VTKHDF/Steps/Values");
            if (times.size() != storedIndices.size())
                throw std::runtime_error("Existing time/step metadata is inconsistent");
            for (std::size_t i = 0; i < storedIndices.size(); ++i)
            {
                if (storedIndices[i] < 0 || (i && storedIndices[i] <= storedIndices[i-1]) ||
                    static_cast<std::size_t>(storedIndices[i]) >= engine.Steps() ||
                    times[i] != ReadTime(io, engine, static_cast<std::size_t>(storedIndices[i])))
                    throw std::runtime_error("Existing source steps or times do not match the input");
            }
            if (LinkExists(file, "VTKHDF/FieldData/globalElementIds"))
            {
                hid_t ids = H5Valid(H5Dopen2(file, "VTKHDF/FieldData/globalElementIds", H5P_DEFAULT), "open IDs");
                globalIdCount = DatasetDimensions(ids).front();
                H5Dclose(ids);
            }
            H5Fclose(file); file = -1;
        }
        std::set<std::string> targetSet(existingFields.begin(), existingFields.end());
        const auto requestedFields = RequestedFieldNames(options, availableFields);
        targetSet.insert(requestedFields.begin(), requestedFields.end());
        const std::vector<std::string> targetFields(targetSet.begin(), targetSet.end());
        const auto requestedSteps = RequestedSteps(options, targetFields, availableFields);
        std::vector<std::size_t> allSteps(storedIndices.begin(), storedIndices.end());
        std::set<std::size_t> present(allSteps.begin(), allSteps.end());
        for (auto step : requestedSteps)
        {
            if (present.count(step)) continue;
            if (!allSteps.empty() && step <= allSteps.back())
                throw std::runtime_error("New source steps must append after existing steps");
            allSteps.push_back(step);
            present.insert(step);
        }
        if (allSteps.empty()) throw std::runtime_error("No source steps selected");
        for (auto step : allSteps)
        {
            if (step >= engine.Steps()) throw std::runtime_error("Requested BP step is out of range");
            (void)ReadTime(io, engine, step);
            for (const auto& name : targetFields)
            {
                const auto& field = availableFields.at(name);
                if (field.type == "float") ValidateFieldStep<float>(io, engine, field, mesh, step);
                else ValidateFieldStep<double>(io, engine, field, mesh, step);
            }
        }
        for (const auto& scalar : scalars)
            DispatchScalar(scalar, [&](auto value) {
                (void)ScalarValues<decltype(value)>(io, engine, scalar, allSteps);
            });
        const std::size_t oldCount = storedIndices.size(), newCount = allSteps.size();
        if (outputExists && oldCount == newCount && targetFields.size() == existingFields.size())
        {
            H5Pclose(access);
            engine.Close();
            std::cout << "Nothing to append: all selected fields and steps already exist.\n";
            return 0;
        }

        writePath = options.output;
        if (options.overwrite && fs::exists(options.output)) writePath += ".partial";
        fs::create_directories(writePath.parent_path());
        file = outputExists
            ? H5Valid(H5Fopen(writePath.string().c_str(), H5F_ACC_RDWR, access), "open output")
            : H5Valid(H5Fcreate(writePath.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, access), "create output");
        H5Pclose(access);
        createdOutput = !outputExists;
        hid_t root, pointData, fieldData, steps, pointOffsets, fieldOffsets, fieldSizes, state;
        if (!outputExists)
        {
            root = CreateGroup(file, "VTKHDF");
            WriteStringAttribute(root, "Type", "UnstructuredGrid");
            WriteAttribute<std::int64_t>(root, "Version", {2, 3});
            pointData = CreateGroup(root, "PointData");
            fieldData = CreateGroup(root, "FieldData");
            WriteSmallDataset<std::int64_t>(root, "NumberOfPoints", {static_cast<std::int64_t>(mesh.numberOfPoints)});
            WriteSmallDataset<std::int64_t>(root, "NumberOfCells", {static_cast<std::int64_t>(mesh.numberOfCells)});
            WriteSmallDataset<std::int64_t>(root, "NumberOfConnectivityIds", {static_cast<std::int64_t>(mesh.numberOfCells * 8)});
            std::cout << "Writing shared geometry...\n";
            if (mesh.type == "float") WriteGeometry<float>(io, engine, root, mesh, options.chunkTuples);
            else WriteGeometry<double>(io, engine, root, mesh, options.chunkTuples);
            globalIdCount = WriteStaticFieldData(io, engine, fieldData, mesh, order, options.chunkTuples);
            steps = CreateGroup(root, "Steps");
            WriteAttribute<std::int64_t>(steps, "NSteps", {0});
            pointOffsets = CreateGroup(steps, "PointDataOffsets");
            fieldOffsets = CreateGroup(steps, "FieldDataOffsets");
            fieldSizes = CreateGroup(steps, "FieldDataSizes");
            state = CreateGroup(file, STATE_GROUP);
            WriteAttribute<std::int64_t>(state, "NativeSchemaVersion", {2});
            WriteStringAttribute(state, "Writer", "bp2vtkhdf");
            WriteStringAttribute(state, "SourceBP", fs::canonical(options.input).string());
            WriteSmallDataset<std::uint64_t>(state, "PointCounts", {mesh.pointCounts.begin(), mesh.pointCounts.end()});
            WriteSmallDataset<std::uint64_t>(state, "WriterIds", {mesh.writers.begin(), mesh.writers.end()});
        }
        else
        {
            root = OpenGroup(file, "VTKHDF");
            pointData = OpenGroup(root, "PointData");
            fieldData = OpenGroup(root, "FieldData");
            steps = OpenGroup(root, "Steps");
            pointOffsets = OpenGroup(steps, "PointDataOffsets");
            fieldOffsets = OpenGroup(steps, "FieldDataOffsets");
            fieldSizes = OpenGroup(steps, "FieldDataSizes");
            state = OpenGroup(file, STATE_GROUP);
        }
        SetAttribute<std::int64_t>(state, "InProgress", {1});
        H5Check(H5Fflush(file, H5F_SCOPE_GLOBAL), "publish write-in-progress flag");
        std::cout << "Mesh: " << mesh.numberOfPoints << " points, " << mesh.numberOfCells
                  << " cells, " << mesh.pointCounts.size() << " blocks; steps " << oldCount << " -> " << newCount << '\n';
        const std::set<std::string> existingSet(existingFields.begin(), existingFields.end());
        for (const auto& name : targetFields)
        {
            const std::size_t first = existingSet.count(name) ? oldCount : 0;
            std::vector<std::size_t> positions, sourceSteps;
            for (std::size_t pos = first; pos < newCount; ++pos)
            {
                positions.push_back(pos); sourceSteps.push_back(allSteps[pos]);
            }
            if (positions.empty()) continue;
            const auto& field = availableFields.at(name);
            if (field.type == "float")
                WritePointFieldPositions<float>(io, engine, pointData, pointOffsets, field, mesh,
                                                sourceSteps, positions, newCount, options.chunkTuples);
            else
                WritePointFieldPositions<double>(io, engine, pointData, pointOffsets, field, mesh,
                                                 sourceSteps, positions, newCount, options.chunkTuples);
        }
        WriteScalarMetadata(io, engine, fieldData, fieldOffsets, fieldSizes, scalars, allSteps);
        UpdateStepMetadata(io, engine, steps, state, allSteps, oldCount);
        UpdateFieldDataStepMetadata(fieldOffsets, fieldSizes, oldCount, newCount, globalIdCount);
        SetAttribute<std::int64_t>(steps, "NSteps", {static_cast<std::int64_t>(newCount)});
        SetAttribute<std::int64_t>(state, "InProgress", {0});
        H5Check(H5Fflush(file, H5F_SCOPE_GLOBAL), "flush output");
        H5Check(H5Fclose(file), "close output"); file = -1;
        engine.Close();
        if (writePath != options.output)
        {
#ifdef _WIN32
            fs::remove(options.output);
#endif
            fs::rename(writePath, options.output);
        }
        createdOutput = false;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "Saved " << options.output << " in " << seconds << " s\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        if (file >= 0) H5Fclose(file);
        if (dynamic_cast<const std::invalid_argument*>(&error)) PrintUsage(argc > 0 ? argv[0] : "bp2vtkhdf");
        std::cerr << "ERROR: " << error.what() << '\n';
        if (createdOutput && !writePath.empty())
        {
            std::error_code ignored;
            fs::remove(writePath, ignored);
        }
        return 1;
    }
}
